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

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
n_epochs = 500

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
model_with_uncertainty.optimize_hyperparameters(valid_dataloader)

import tqdm

estimated_variances = []
actual_variances = []
with torch.no_grad():
    for batch in tqdm.tqdm(test_dataloader):
        input, result = batch 
        prediction, uncertainty = model_with_uncertainty(input)
        estimated_variances.append(uncertainty)
        actual_variances.append(
            (prediction-result)**2
        )
actual_variances = torch.concatenate(actual_variances).squeeze(1)
estimated_variances = torch.concatenate(estimated_variances).squeeze(1)
actual_variances = actual_variances.cpu().numpy()
estimated_variances = estimated_variances.cpu().numpy()

np.save(f"outputs/figures/estimated_variances_{n_exp}.npy", estimated_variances)
np.save(f"outputs/figures/actual_variances_{n_exp}.npy", actual_variances)
