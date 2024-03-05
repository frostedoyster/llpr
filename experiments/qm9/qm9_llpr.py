# Since the atomistic libraries are not very interoperable with
# torch workflows, we need to re-write the llpr model to support them

import copy
from typing import Tuple

import torch

from llpr.utils.metrics import avg_nll_regression
from llpr.utils.validation_opt import validation_opt


def _hook(module, input: Tuple[torch.Tensor], output) -> Tuple[torch.Tensor, torch.Tensor]:
    return output, input[0]

class UncertaintyModel(torch.nn.Module):
    def __init__(self, model, last_layer, train_loader=None, exponent=0.0):
        super().__init__()

        # do not deepcopy: the rascaline.torch calculator can't handle it
        self.model = model.eval()

        if isinstance(last_layer, str):
            last_layer = self.model._modules[last_layer]
        if not isinstance(last_layer, torch.nn.Module):
            raise TypeError("The last layer should be a torch.nn.Module")
        if not isinstance(last_layer, torch.nn.Linear):
            raise TypeError("The last layer should be a linear layer")
        self.last_layer_has_bias = last_layer.bias is not None

        # Get the name of the last layer
        for name, module in model.named_modules():
            if module is last_layer:
                last_layer_name = name
                break

        # Register a forward hook on the last layer
        self.model._modules[last_layer_name].register_forward_hook(_hook)

        self.hidden_size = last_layer.in_features + self.last_layer_has_bias
        self.register_buffer("covariance", torch.zeros((self.hidden_size, self.hidden_size), device=next(self.model.parameters()).device))
        self.register_buffer("sigma", torch.zeros((), device=next(self.model.parameters()).device))
        self.register_buffer("inv_covariance", torch.zeros((self.hidden_size, self.hidden_size), device=next(self.model.parameters()).device))
        self.has_covariance = False
        self.hypers_are_set = False

        if train_loader is None:
            # this is useful for loading a saved model from a state dict
            # TODO: change the design. This is not very elegant.
            # You could split a set_model from the constructor
            self.has_covariance = True
            self.hypers_are_set = True
            return

        # TODO: check that uncertainty layers is not empty,
        # that all requested layers are part of the model and so on

        with torch.no_grad():
            for batch in train_loader:
                x, y = batch
                # x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                y_predicted, hidden_features = self.model(x)
                n_atoms = torch.tensor([len(structures.positions) for structures in x], device=next(model.parameters()).device).unsqueeze(1)
                hidden_features = hidden_features / (n_atoms**exponent)
                if self.last_layer_has_bias:
                    hidden_features = torch.cat((hidden_features, torch.ones_like(hidden_features[:, :1])), dim=1)
                self.covariance += hidden_features.T @ hidden_features
        self.has_covariance = True

    def forward(self, x):
        x = list(x)  # make sure we have a list, and not a tuple for example
        prediction, hidden_features = self.model(x)
        if self.last_layer_has_bias:
            hidden_features = torch.cat((hidden_features, torch.ones_like(hidden_features[:, :1])), dim=1)
        uncertainty = torch.einsum("ij, jk, ik -> i", hidden_features, self.inv_covariance, hidden_features).unsqueeze(1)
        return prediction, uncertainty

    def set_hyperparameters(self, C, sigma):
        if not self.has_covariance:
            raise RuntimeError("The convariance has not been calculated. Please pass a training loader to the constructor.")
        
        # Utility function to set the hyperparameters of the uncertainty model.
        C = float(C)
        sigma = float(sigma)
        self.inv_covariance = C * torch.linalg.inv(self.covariance + sigma**2 * torch.eye(self.hidden_size, device=self.covariance.device))
        self.hypers_are_set = True

    def optimize_hyperparameters(self, validation_dataloader, device=None):
        validation_opt(self, validation_dataloader, avg_nll_regression, 2, device=device)
