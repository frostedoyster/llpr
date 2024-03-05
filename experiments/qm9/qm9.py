import numpy as np
import torch

from dataset import get_dataset
from model import FeatureCalculator, AtomicNNs, SumAtomsModule
from train import train_model
from qm9_llpr import UncertaintyModel

print("Start")

import sys
n_exp = int(sys.argv[1])

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
n_epochs = 300

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

loss_fn = torch.nn.MSELoss()  # mean square error
optimizer = torch.optim.AdamW(model.parameters())

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

train_model(model, optimizer, loss_fn, train_dataloader, valid_dataloader, n_epochs, n_exp, device)

torch.save(model.state_dict(), f"outputs/models/qm9_{n_exp}.pt")

model_with_uncertainty = UncertaintyModel(model, model[-1], train_dataloader, n_exp)
model_with_uncertainty.optimize_hyperparameters(valid_dataloader)

import tqdm

estimated_errors = []
actual_errors = []
with torch.no_grad():
    for batch in tqdm.tqdm(test_dataloader):
        input, result = batch 
        prediction, uncertainty = model_with_uncertainty(input)
        estimated_errors.append(uncertainty)
        actual_errors.append(
            (prediction-result)**2
        )
actual_errors = torch.concatenate(actual_errors).squeeze(1)
estimated_errors = torch.concatenate(estimated_errors).squeeze(1)
actual_errors = actual_errors.cpu().numpy()
estimated_errors = estimated_errors.cpu().numpy()

n_samples_per_bin = 100
n_sigma = 1.0

def split_array(array, size):
    n_splits = len(array) // size
    split_array = []
    for i_split in range(n_splits):
        split_array.append(
            array[i_split*size:(i_split+1)*size]
        )
    return split_array

sort_indices = np.argsort(estimated_errors)
estimated_errors = estimated_errors[sort_indices]
actual_errors = actual_errors[sort_indices]

split_estimated = split_array(estimated_errors, n_samples_per_bin)
split_actual = split_array(actual_errors, n_samples_per_bin)

estimated_averages = np.array([np.mean(split_estimated_single) for split_estimated_single in split_estimated])
actual_averages = np.array([np.mean(split_actual_single) for split_actual_single in split_actual])

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

max_value = max(np.max(estimated_averages), np.max(actual_averages))
min_value = min(np.min(estimated_averages), np.min(actual_averages))

plt.plot([min_value, max_value], [min_value, max_value], "k--", label="y=x")
plt.plot(estimated_averages, actual_averages, ".", markersize=4.0)
# plt.fill_between([0.0, np.max(estimated_averages)], [0.0, np.max(estimated_averages)+n_sigma*np.sqrt(2.0/n_samples_per_bin)*np.max(estimated_averages)], [0.0, np.max(estimated_averages)-n_sigma*np.sqrt(2.0/n_samples_per_bin)*np.max(estimated_averages)], alpha=0.3, color="tab:orange")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Predicted variance $(kcal^2/mol^2)$")
plt.ylabel(r"Actual variance $(kcal^2/mol^2)$")
plt.tight_layout()
plt.savefig(f"outputs/figures/qm9_{n_exp}.pdf")
