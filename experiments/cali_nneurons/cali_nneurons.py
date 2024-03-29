from sklearn.datasets import fetch_california_housing

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)


data = fetch_california_housing()
features, targets = data.data, data.target
features = torch.tensor(features, dtype=torch.get_default_dtype())
targets = torch.tensor(targets, dtype=torch.get_default_dtype())
number_of_data = targets.shape[0]
permutation = torch.tensor(np.random.permutation(number_of_data), dtype=torch.int64)
features = features[permutation]
targets = targets[permutation]

n_train = 2 * number_of_data // 10
n_valid = 4 * number_of_data // 10
n_test = number_of_data - n_train - n_valid

X_train = features[:n_train]
X_valid = features[n_train:n_train+n_valid]
X_test = features[n_train+n_valid:]
y_train = targets[:n_train].reshape(-1, 1)
y_valid = targets[n_train:n_train+n_valid].reshape(-1, 1)
y_test = targets[n_train+n_valid:].reshape(-1, 1)

mean_X = torch.mean(X_train, dim=0)
std_X = torch.std(X_train, dim=0, correction=0)
std_X[std_X == 0.0] = 1.0  # avoid division by zero
X_train = (X_train - mean_X) / std_X
X_valid = (X_valid - mean_X) / std_X
X_test = (X_test - mean_X) / std_X

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
n_layers = 2

# Set the neurons per layer from outside:
import sys
n_neurons_per_layer = int(sys.argv[1])

model = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], n_neurons_per_layer),
    torch.nn.SiLU(),
    torch.nn.Linear(n_neurons_per_layer, n_neurons_per_layer),
    torch.nn.SiLU(),
    torch.nn.Linear(n_neurons_per_layer, 1, bias=False),
)

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

n_epochs = 400
loss_fn = torch.nn.functional.mse_loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

from llpr import UncertaintyModel
from llpr.utils.train_tensor_inputs import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
train_model(model, optimizer, loss_fn, train_dataloader, valid_dataloader, n_epochs, device)

# save
torch.save(model.state_dict(), f"outputs/models/cali_ood_{n_neurons_per_layer}.pt")

model_with_uncertainty = UncertaintyModel(model, model[-1], train_dataloader)
model_with_uncertainty.optimize_hyperparameters(valid_dataloader, device=device)

def get_estimated_and_actual_variances(dataloader):
    estimated_variances = []
    actual_variances = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, var_pred = model_with_uncertainty(X_batch)
            actual_variances.append((y_pred - y_batch) ** 2)
            estimated_variances.append(var_pred)
    estimated_variances = torch.cat(estimated_variances, dim=0).squeeze(1)
    actual_variances = torch.cat(actual_variances, dim=0).squeeze(1)

    estimated_variances = estimated_variances.cpu().numpy()
    actual_variances = actual_variances.cpu().numpy()

    return estimated_variances, actual_variances

estimated_variances_valid, actual_variances_valid = get_estimated_and_actual_variances(valid_dataloader)
estimated_variances_test, actual_variances_test = get_estimated_and_actual_variances(test_dataloader)

n_samples_per_bin = 100
n_sigma = 1

sorting = np.argsort(estimated_variances_valid)
estimated_variances_valid = estimated_variances_valid[sorting]
actual_variances_valid = actual_variances_valid[sorting]

bins = np.arange(0, len(estimated_variances_valid), n_samples_per_bin)
estimated_variances_valid_avg = np.array([np.mean(estimated_variances_valid[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_valid_avg = np.array([np.mean(actual_variances_valid[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Exclude the last bin, which will most likely be incomplete:
estimated_variances_valid_avg = estimated_variances_valid_avg[:-1]
actual_variances_valid_avg = actual_variances_valid_avg[:-1]

sorting = np.argsort(estimated_variances_test)
estimated_variances_test = estimated_variances_test[sorting]
actual_variances_test = actual_variances_test[sorting]

bins = np.arange(0, len(estimated_variances_test), n_samples_per_bin)
estimated_variances_test_avg = np.array([np.mean(estimated_variances_test[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_test_avg = np.array([np.mean(actual_variances_test[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Exclude the last bin, which will most likely be incomplete:
estimated_variances_test_avg = estimated_variances_test_avg[:-1]
actual_variances_test_avg = actual_variances_test_avg[:-1]

from llpr.utils.metrics import avg_nll_regression

average_nll = avg_nll_regression(model_with_uncertainty, test_dataloader, device=device)
print(f"Average NLL: {average_nll}")

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

min_value = min(np.min(estimated_variances_test_avg), np.min(actual_variances_test_avg))
max_value = max(np.max(estimated_variances_test_avg), np.max(actual_variances_test_avg))

plt.plot([min_value, max_value], [min_value, max_value], "k--", label="y=x")
print(estimated_variances_valid_avg.shape)
print(actual_variances_valid_avg.shape)
print(estimated_variances_test_avg.shape)
print(actual_variances_test_avg.shape)

# plt.plot(estimated_variances_valid_avg, actual_variances_valid_avg, "o")
plt.plot(estimated_variances_test_avg, actual_variances_test_avg, "o")
plt.text(0.95, 0.05, f'avg. NLL={average_nll:.3f}', verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes)
# plt.fill_between([0.0, np.max(estimated_variances_test_avg)], [0.0, np.max(estimated_variances_test_avg)+n_sigma*np.sqrt(2.0/n_samples_per_bin)*np.max(estimated_variances_test_avg)], [0.0, np.max(estimated_variances_test_avg)-n_sigma*np.sqrt(2.0/n_samples_per_bin)*np.max(estimated_variances_test_avg)], alpha=0.3, color="tab:orange")
plt.xscale("log")
plt.yscale("log")
plt.title(f"{n_neurons_per_layer} neurons per layer")
plt.xlabel("Estimated variance")
plt.ylabel("Actual variance")
plt.tight_layout()
plt.savefig(f"outputs/figures/cali_{n_neurons_per_layer}_nll.pdf")
