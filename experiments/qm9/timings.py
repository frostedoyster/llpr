import numpy as np
import torch

from dataset import get_dataset
from model import FeatureCalculator, AtomicNNs, SumAtomsModule
from train import train_model
from qm9_llpr import UncertaintyModel

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

import sys
n_exp = int(sys.argv[1])

device = "cuda"
batch_size = 8

# Define the model
n_neurons = 64
n_neurons_last_layer = 64
activation = torch.nn.SiLU()
all_species = [1, 6, 7, 8, 9]
n_species = len(all_species)
l_max = 4
n_max = 6
n_features = (l_max+1) * n_species**2 * n_max**2

model = torch.nn.Sequential(
    FeatureCalculator(all_species, l_max, n_max),
    AtomicNNs(all_species, n_features, n_neurons, n_neurons_last_layer, activation),
    SumAtomsModule(all_species, n_neurons_last_layer),
    torch.nn.Linear(n_species*n_neurons_last_layer, 1, bias=False),
).to(device)

train_dataset, valid_dataset, test_dataset = get_dataset()

def collate_fn(structures_and_targets):
    structures = []
    targets = []
    for structure, target in structures_and_targets:
        structures.append(structure)
        targets.append(target)  
    return structures, torch.tensor(targets, device=device).unsqueeze(1) * 627.509 # convert from Hartree to kcal/mol

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

state_dict = torch.load(f"outputs/models/qm9_{n_exp}.pt")
model.load_state_dict(state_dict)

model_with_uncertainty = UncertaintyModel(model, model[-1], train_dataloader, n_exp)
model_with_uncertainty.set_hyperparameters(0.1, 0.1)  # won't matter for timings

import time
import tqdm

total_time_raw = 0.0
for X_batch, y_batch in tqdm.tqdm(test_dataloader):
    start = time.time()
    y_pred = model(X_batch)
    torch.cuda.synchronize()
    end = time.time()
    total_time_raw += end - start
print("Raw model: ", total_time_raw)

total_time_uncertainty = 0.0
for X_batch, y_batch in tqdm.tqdm(test_dataloader):
    start = time.time()
    y_pred, var_pred = model_with_uncertainty(X_batch)
    torch.cuda.synchronize()
    end = time.time()
    total_time_uncertainty += end - start
print("Model with uncertainty: ", total_time_uncertainty)
