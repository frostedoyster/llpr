import pandas as pd
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

device = "cuda"

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

n_neurons = 256

n_train = 4 * number_of_data // 10
n_valid = 3 * number_of_data // 10

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

batch_size = 8

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# we don't train: doesn't matter for timings

from llpr import UncertaintyModel

model_with_uncertainty = UncertaintyModel(model, model[-1], train_dataloader)
model_with_uncertainty.set_hyperparameters(0.1, 0.1)  # doesn't matter for timings

import time
import tqdm

total_time_raw = 0.0
for l_batch, X_batch, y_batch in tqdm.tqdm(test_dataloader):
    l_batch, X_batch, y_batch = l_batch.to(device), X_batch.to(device), y_batch.to(device)
    start = time.time()
    y_pred = model((l_batch, X_batch))
    torch.cuda.synchronize()
    end = time.time()
    total_time_raw += end - start
print("Raw model: ", total_time_raw)

total_time_uncertainty = 0.0
for l_batch, X_batch, y_batch in tqdm.tqdm(test_dataloader):
    l_batch, X_batch, y_batch = l_batch.to(device), X_batch.to(device), y_batch.to(device)
    start = time.time()
    y_pred, var_pred = model_with_uncertainty((l_batch, X_batch))
    torch.cuda.synchronize()
    end = time.time()
    total_time_uncertainty += end - start
print("Model with uncertainty: ", total_time_uncertainty)
