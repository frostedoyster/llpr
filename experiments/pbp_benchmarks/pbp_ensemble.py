import sys

import numpy as np
import torch

from llpr.utils.train_tensor_inputs import train_model
from llpr.deep_ensembles.regression import DeepEnsemble, EnsembleLastLayer, nll_loss, train_nll
from llpr.deep_ensembles.train import train_ensemble
from llpr.utils.metrics import avg_nll_regression, rmse, regression_uncertainty_parity_plot

from dataset import get_dataset


seed = int(sys.argv[2])
torch.manual_seed(seed)
np.random.seed(seed)

dataset_name = sys.argv[1]
train_dataset, valid_dataset, test_dataset, mean_y, std_y = get_dataset(dataset_name)

n_epochs = 400
batch_size = 32
n_ensemble = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

n_features = train_dataset.tensors[0].shape[1]
n_neurons = 100 if (dataset_name == "protein" or dataset_name == "year") else 50

models = [
    torch.nn.Sequential(
        torch.nn.Linear(n_features, n_neurons),
        torch.nn.ReLU(),
        EnsembleLastLayer(n_neurons)
    )
    for _ in range(n_ensemble)
]

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizers = [torch.optim.Adam(model.parameters()) for model in models]
for model in models:
    model.to(device)
train_ensemble(train_nll, models, optimizers, nll_loss, train_dataloader, valid_dataloader, n_epochs, device)
model_with_uncertainty = DeepEnsemble(models)

torch.save(model.state_dict(), f"outputs/models/ensemble_{dataset_name}_{seed}.pt")

print("nll:", avg_nll_regression(model_with_uncertainty, test_dataloader, device, mean_y, std_y))
print("rmse:", rmse(model_with_uncertainty, test_dataloader, device, mean_y, std_y))
regression_uncertainty_parity_plot(model_with_uncertainty, test_dataloader, device, f"outputs/figures/ensemble_{dataset_name}_{seed}.pdf", None, mean_y, std_y)
