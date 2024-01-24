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

n_train = 4 * number_of_data // 10
n_valid = number_of_data // 10
n_test = number_of_data - n_train

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

import matplotlib.pyplot as plt

plt.figure()
plt.hist(X_train[:, 7], bins=50)
plt.title('Histogram of the 8th feature')
plt.savefig("outputs/figures/ood_cali_hist.pdf")
plt.close()

in_domain_indices_train = (X_train[:, 7] > -0.3).nonzero().squeeze()
print(in_domain_indices_train)
X_train = X_train[in_domain_indices_train]
y_train = y_train[in_domain_indices_train]

in_domain_indices_valid = (X_valid[:, 7] > -0.3).nonzero().squeeze()
print(in_domain_indices_valid)
X_valid = X_valid[in_domain_indices_valid]
y_valid = y_valid[in_domain_indices_valid]

in_domain_indices_test = (X_test[:, 7] > -0.3).nonzero().squeeze()
out_of_domain_indices_test = (X_test[:, 7] <= -0.3).nonzero().squeeze()
print(in_domain_indices_test)
X_test_id = X_test[in_domain_indices_test]
y_test_id = y_test[in_domain_indices_test]
X_test_ood = X_test[out_of_domain_indices_test]
y_test_ood = y_test[out_of_domain_indices_test]

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test_id, y_test_id)
ood_test_dataset = torch.utils.data.TensorDataset(X_test_ood, y_test_ood)

n_layers = 2
n_neurons_per_layer = 32

model = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], n_neurons_per_layer),
    torch.nn.SiLU(),
    torch.nn.Linear(n_neurons_per_layer, n_neurons_per_layer),
    torch.nn.SiLU(),
    torch.nn.Linear(n_neurons_per_layer, 1),
)

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))
print(len(ood_test_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
ood_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=128, shuffle=False)

n_epochs = 400
loss_fn = torch.nn.functional.mse_loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

from llpr import UncertaintyModel
from llpr.utils.train_tensor_inputs import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
train_model(model, optimizer, loss_fn, train_dataloader, valid_dataloader, n_epochs, device)


# with torch.no_grad():
#     y_pred = model(X_valid.to(device))
#     valid_loss = torch.nn.functional.mse_loss(y_pred, y_valid.to(device)).item()
#     print(f"Before training: Valid loss: {valid_loss:.4f}")

# best_model = None
# best_valid_loss = float("inf")
# for epoch in range(n_epochs):
#     for X_batch, y_batch in train_dataloader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         y_pred = model(X_batch)
#         loss = torch.nn.functional.mse_loss(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch}")

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
    estimated_variances = torch.cat(estimated_variances, dim=0)
    actual_variances = torch.cat(actual_variances, dim=0)

    estimated_variances = estimated_variances.cpu().numpy()
    actual_variances = actual_variances.cpu().numpy()

    return estimated_variances, actual_variances

estimated_variances_valid, actual_variances_valid = get_estimated_and_actual_variances(valid_dataloader)
estimated_variances_test, actual_variances_test = get_estimated_and_actual_variances(test_dataloader)
estimated_variances_ood, actual_variances_ood = get_estimated_and_actual_variances(ood_test_dataloader)

n_per_bin = 100

sorting = np.argsort(estimated_variances_valid)
estimated_variances_valid = estimated_variances_valid[sorting]
actual_variances_valid = actual_variances_valid[sorting]

bins = np.arange(0, len(estimated_variances_valid), n_per_bin)
estimated_variances_valid_avg = np.array([np.mean(estimated_variances_valid[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_valid_avg = np.array([np.mean(actual_variances_valid[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Warn about the fact that the last bin is not full:
print(f"Other bins contain {n_per_bin} elements.")
print(f"The last bin contains only {len(estimated_variances_valid) - bins[-1]} elements.")

sorting = np.argsort(estimated_variances_test)
estimated_variances_test = estimated_variances_test[sorting]
actual_variances_test = actual_variances_test[sorting]

bins = np.arange(0, len(estimated_variances_test), n_per_bin)
estimated_variances_test_avg = np.array([np.mean(estimated_variances_test[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_test_avg = np.array([np.mean(actual_variances_test[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Warn about the fact that the last bin is not full:
print(f"Other bins contain {n_per_bin} elements.")
print(f"The last bin contains only {len(estimated_variances_test) - bins[-1]} elements.")

sorting = np.argsort(estimated_variances_ood)
estimated_variances_ood = estimated_variances_ood[sorting]
actual_variances_ood = actual_variances_ood[sorting]

bins = np.arange(0, len(estimated_variances_ood), n_per_bin)
estimated_variances_ood_avg = np.array([np.mean(estimated_variances_ood[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_ood_avg = np.array([np.mean(actual_variances_ood[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Warn about the fact that the last bin is not full:
print(f"Other bins contain {n_per_bin} elements.")
print(f"The last bin contains only {len(estimated_variances_ood) - bins[-1]} elements.")

import matplotlib.pyplot as plt

min_value = min(np.min(estimated_variances_test_avg), np.min(actual_variances_test_avg), np.min(estimated_variances_ood_avg), np.min(actual_variances_ood_avg))
max_value = max(np.max(estimated_variances_test_avg), np.max(actual_variances_test_avg), np.max(estimated_variances_ood_avg), np.max(actual_variances_ood_avg))

plt.plot([min_value, max_value], [min_value, max_value], "k--")
print(estimated_variances_valid_avg.shape)
print(actual_variances_valid_avg.shape)
print(estimated_variances_test_avg.shape)
print(actual_variances_test_avg.shape)
print(estimated_variances_ood_avg.shape)
print(actual_variances_ood_avg.shape)
# plt.plot(estimated_variances_valid_avg, actual_variances_valid_avg, "o")
plt.plot(estimated_variances_test_avg, actual_variances_test_avg, "o")
plt.plot(estimated_variances_ood_avg, actual_variances_ood_avg, "o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Estimated variance")
plt.ylabel("Actual variance")
plt.savefig("outputs/figures/ood_cali.pdf")
