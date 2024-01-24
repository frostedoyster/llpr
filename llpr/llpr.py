import copy
from typing import Tuple

import torch

from llpr.utils.metrics import avg_nll_regression, avg_nll_classification
from llpr.utils.validation_opt import validation_opt


def _hook(module, input: Tuple[torch.Tensor], output) -> Tuple[torch.Tensor, torch.Tensor]:
    return output, input[0]

class UncertaintyModel(torch.nn.Module):
    def __init__(self, model, last_layer, train_loader=None, mode="classification"):
        super().__init__()
        self.mode = mode
        if mode == "classification":
            self.n_samples = 1000  # hardcoded for now. TODO: make this a parameter
        else:
            self.n_samples = -1

        # we are going to register a forward hook on the last layer
        # and we don't want to modify the original model, so the user
        # can keep using it after creating the uncertainty model
        self.model = copy.deepcopy(model).eval()

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
                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                y_predicted, hidden_features = self.model(x)
                if self.last_layer_has_bias:
                    hidden_features = torch.cat((hidden_features, torch.ones_like(hidden_features[:, :1])), dim=1)
                self.covariance += hidden_features.T @ hidden_features
        self.has_covariance = True

    def forward(self, x):
        if not self.has_covariance:
            raise RuntimeError("The convariance has not been calculated. Please pass a training loader to the constructor.")
        if not self.hypers_are_set:
            raise RuntimeError("The hyperparameters have not been set. Please call set_hyperparameters or optimize_hyperparameters.")
        prediction, hidden_features = self.model(x)
        if self.last_layer_has_bias:
            hidden_features = torch.cat((hidden_features, torch.ones_like(hidden_features[:, :1])), dim=1)
        # Using torch.linalg.solve:
        # uncertainty = (torch.linalg.solve(self.covariance, hidden_features.T).T * hidden_features).sum(dim=1, keepdim=True)
        uncertainty = torch.einsum("ij, jk, ik -> i", hidden_features, self.inv_covariance, hidden_features)
        if self.mode == "regression":
            return prediction, uncertainty
        elif self.mode == "classification":
            return prediction, uncertainty
            # means = prediction
            # covariances = uncertainty.repeat(1, means.shape[1])
            # stds = torch.sqrt(covariances)  # calculate standard deviations
            # sampler = torch.distributions.Normal(means, stds)
            # samples = sampler.sample((self.n_samples,))
            # # Average argmax over samples
            # bool_argmax = torch.argmax(samples, dim=-1, keepdim=True) == torch.arange(means.shape[1], device=samples.device).unsqueeze(0).unsqueeze(0)
            # result = torch.mean(bool_argmax.to(samples.dtype), dim=0)

            # # # Average probabilities over samples
            # # samples = torch.softmax(samples, dim=-1)
            # # result = torch.mean(samples, dim=0)  # average probabilities over samples
            # return result
        else:
            raise RuntimeError(f"Invalid mode: {self.mode}.")

    def set_hyperparameters(self, C, sigma):
        if not self.has_covariance:
            raise RuntimeError("The convariance has not been calculated. Please pass a training loader to the constructor.")
        
        # Utility function to set the hyperparameters of the uncertainty model.
        C = float(C)
        sigma = float(sigma)
        self.inv_covariance = C * torch.linalg.inv(self.covariance + sigma**2 * torch.eye(self.hidden_size, device=self.covariance.device, dtype=torch.float64)).to(self.covariance.dtype)
        # torch.cholesky_inverse(self.covariance.to(torch.float64) + sigma**2 * torch.eye(self.hidden_size, dtype=torch.float64, device=self.covariance.device)).to(self.covariance.dtype)
        self.hypers_are_set = True

    def optimize_hyperparameters(self, validation_dataloader, device=None):
        if self.mode == "regression":
            validation_opt(self, validation_dataloader, avg_nll_regression, 2, device=device)
        elif self.mode == "classification":
            validation_opt(self, validation_dataloader, avg_nll_classification, 2, device=device)
        else:
            raise RuntimeError(f"Invalid mode: {self.mode}.")