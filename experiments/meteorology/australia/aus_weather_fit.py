import pandas as pd
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

data = pd.read_csv('data/weatherAUS_processed.csv')
locations, features, targets = data['Location'], data.drop(['Location', 'NextDayMaxTemp'], axis=1), data['NextDayMaxTemp']

locations = torch.tensor(locations, dtype=torch.int64)
features = torch.tensor(features.values, dtype=torch.get_default_dtype())
targets = torch.tensor(targets.values, dtype=torch.get_default_dtype())
number_of_data = targets.shape[0]
permutation = torch.tensor(np.random.permutation(number_of_data), dtype=torch.int64)
locations = locations[permutation]
features = features[permutation]
targets = targets[permutation]

import sys
n_neurons = int(sys.argv[1])
training_set_fraction = int(sys.argv[2])

valid_set_fraction = (10 - training_set_fraction) // 2

n_train = training_set_fraction * number_of_data // 10
n_valid = valid_set_fraction * number_of_data // 10

l_train = locations[:n_train]
l_valid = locations[n_train:n_train+n_valid]
l_test = locations[n_train+n_valid:]
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

train_dataset = torch.utils.data.TensorDataset(l_train, X_train, y_train)
valid_dataset = torch.utils.data.TensorDataset(l_valid, X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(l_test, X_test, y_test)

n_layers = 2
n_features = X_train.shape[1]

n_neurons_embedding = n_neurons
n_neurons_per_layer = n_neurons

class FirstLayer(torch.nn.Module):
    def __init__(self, n_neurons_embedding):
        super().__init__()
        self.location_embedding = torch.nn.Embedding(26, n_neurons_embedding)

    def forward(self, loc_x):
        loc, x = loc_x
        return torch.cat(
            (self.location_embedding(loc), x),
            dim=1
        )

model = torch.nn.Sequential(
    FirstLayer(n_neurons_embedding),
    torch.nn.Linear(n_neurons_embedding+n_features, n_neurons_per_layer),
    torch.nn.SiLU(),
    torch.nn.Linear(n_neurons_per_layer, n_neurons_per_layer),
    torch.nn.SiLU(),
    torch.nn.Linear(n_neurons_per_layer, 1),
)
model = model.to(device)

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

with torch.no_grad():
    y_pred = model((l_valid.to(device), X_valid.to(device)))
    valid_loss = torch.nn.functional.mse_loss(y_pred, y_valid.to(device))
    print(f"Before training: Valid loss: {valid_loss:.4f}")

from llpr.utils.train_tensor_inputs import train_model

train_model(model, optimizer, torch.nn.functional.mse_loss, train_dataloader, valid_dataloader, 100, device)

from llpr import UncertaintyModel

model_with_uncertainty = UncertaintyModel(model, model[-1], train_dataloader)
model_with_uncertainty.optimize_hyperparameters(valid_dataloader, device)

# covariance = 0.0001*features_last_layer.T @ features_last_layer + 0.5 * torch.eye(features_last_layer.shape[1], device=features_last_layer.device)

estimated_variances = []
actual_variances = []
with torch.no_grad():
    for l_batch, X_batch, y_batch in test_dataloader:
        y_pred, y_unc = model_with_uncertainty((l_batch.to(device), X_batch.to(device)))
        actual_variances.append((y_pred - y_batch.to(device)) ** 2)
        estimated_variances.append(y_unc)
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

# Remove the last bin if it is not full
if len(estimated_variances) % n_per_bin != 0:
    estimated_variances_avg = estimated_variances_avg[:-1]
    actual_variances_avg = actual_variances_avg[:-1]

import matplotlib.pyplot as plt

min_value = min(np.min(estimated_variances_avg), np.min(actual_variances_avg))
max_value = max(np.max(estimated_variances_avg), np.max(actual_variances_avg))
plt.plot([min_value, max_value], [min_value, max_value], "k--")

plt.plot(estimated_variances_avg, actual_variances_avg, "o")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Estimated variance")
plt.ylabel("Actual variance")
plt.savefig(f"outputs/figures/weatherAUS_{n_neurons}_{training_set_fraction}.pdf")
