import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


features = torch.tensor(np.load("data/X.npy"))
targets = torch.tensor(np.load("data/y.npy"))

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
n_neurons_per_layer = 128

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

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

from llpr.utils.train_tensor_inputs import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
train_model(model, optimizer, torch.nn.functional.mse_loss, train_dataloader, valid_dataloader, 100, device)

from llpr import UncertaintyModel

model_with_uncertainty = UncertaintyModel(model, model[-1], train_dataloader)
model_with_uncertainty.optimize_hyperparameters(valid_dataloader, device=device)

from tqdm import tqdm

estimated_variances = []
actual_variances = []
with torch.no_grad():
    for X_batch, y_batch in tqdm(test_dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        actual_variances.append((y_pred - y_batch) ** 2)
        y_pred, var_pred = model_with_uncertainty(X_batch)
        estimated_variances.append(var_pred)
estimated_variances = torch.cat(estimated_variances, dim=0)
actual_variances = torch.cat(actual_variances, dim=0)

estimated_variances = estimated_variances.cpu().numpy()
actual_variances = actual_variances.cpu().numpy()


n_per_bin = 100

sorting = np.argsort(estimated_variances)
estimated_variances = estimated_variances[sorting]
actual_variances = actual_variances[sorting]

bins = np.arange(0, len(estimated_variances), n_per_bin)
estimated_variances_avg = np.array([np.mean(estimated_variances[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_avg = np.array([np.mean(actual_variances[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# eliminate the last bin because it might be incomplete
estimated_variances_avg = estimated_variances_avg[:-1]
actual_variances_avg = actual_variances_avg[:-1]

import matplotlib.pyplot as plt

min_value = min(np.min(estimated_variances_avg), np.min(actual_variances_avg))
max_value = max(np.max(estimated_variances_avg), np.max(actual_variances_avg))

plt.plot(estimated_variances_avg, actual_variances_avg, "o")
plt.plot([min_value, max_value], [min_value, max_value], "k--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Estimated variance")
plt.ylabel("Actual variance")
plt.savefig("lausanne_weather.png")





# In[ ]:




