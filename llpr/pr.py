from math import prod

import torch

from llpr.utils.metrics import avg_nll_regression, rmse
from llpr.utils.validation_opt import validation_opt


class UncertaintyModel(torch.nn.Module):
    # Wraps a model so as to predict targets and uncertainties on the targets.

    def __init__(
        self,
        model,
        uncertainty_layers,
        train_loader=None,
        loss_fn=None
    ):
        super().__init__()

        # make sure the model is in evaluation mode. Any running averages
        # should have reached their a stable value by the end of the training procedure.
        model.eval()

        self.model = model
        uncertainty_parameters = []
        for layer in uncertainty_layers:
            uncertainty_parameters.extend(
                list(layer.parameters())
            )
        self.uncertainty_parameters = uncertainty_parameters
        self.n_uncertainty_parameters = sum([parameter.numel() for parameter in self.uncertainty_parameters])
        self.register_buffer("pseudo_hessian", torch.zeros((self.n_uncertainty_parameters, self.n_uncertainty_parameters), device=next(model.parameters()).device))
        self.register_buffer("sigma", torch.zeros((), device=next(model.parameters()).device))
        self.register_buffer("inv_covariance", torch.zeros((self.n_uncertainty_parameters, self.n_uncertainty_parameters), device=next(model.parameters()).device))
        self.has_pseudo_hessian = False
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

        for batch in train_loader:
            x, y = batch
            x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
            y_predicted = model(x)
            J = _jacobian(
                y_predicted.reshape(len(x), -1),
                self.uncertainty_parameters
            )

            batch_size = len(x)
            assert batch_size == y_predicted.shape[0]
            out_size = prod(y_predicted.shape[1:])
            
            if loss_fn is not None:
                loss_fn_p = lambda a: loss_fn(a, y)
                y_predicted.requires_grad_(True)
                H = torch.autograd.functional.hessian(
                    loss_fn_p,
                    y_predicted,
                    vectorize=True
                )
                assert H.shape == y_predicted.shape + y_predicted.shape
                H = torch.sum(H, dim=0)  # sum over one of the two batch dimensions
                H = H.reshape(out_size, batch_size, out_size)
            else:
                # loss-agnostic Hessian calculation
                # TODO: make more computationally efficient.
                # This is a hack (effectively a matrix multiplication by identity matrices)
                H = torch.stack([torch.eye(out_size, device=J.device)/out_size for _ in range(batch_size)], dim=1)

            for i in range(batch_size):
                J_i = J[i]
                H_i = H[:, i, :]
                self.pseudo_hessian += J_i.T @ H_i @ J_i
            # self.pseudo_hessian += torch.einsum("bow, obp, bpv -> wv", J, H, J)

        self.has_pseudo_hessian = True

    def forward(self, x):
        if not self.hypers_are_set:
            raise RuntimeError("The hyperparameters have not been set. Please call set_hyperparameters.")
        
        predictions = self.model(x)
        J = _jacobian(
            predictions.reshape(len(x), -1),
            self.uncertainty_parameters
        )
        uncertainty_estimates = torch.einsum("bow, wv, bov -> bo", J, self.inv_covariance, J)
        uncertainty_estimates = uncertainty_estimates.reshape(predictions.shape)
        return predictions, uncertainty_estimates

    def set_hyperparameters(self, C, sigma):
        if not self.has_pseudo_hessian:
            raise RuntimeError("The pseudo-hessian has not been calculated. Please pass a training loader to the constructor.")
        
        # Utility function to set the hyperparameters of the uncertainty model.
        C = float(C)
        sigma = float(sigma)
        self.inv_covariance = C * torch.linalg.inv(self.pseudo_hessian + sigma**2 * torch.eye(self.n_uncertainty_parameters, device=self.pseudo_hessian.device))
        self.hypers_are_set = True

    def optimize_hyperparameters(self, validation_dataloader, device=None):
        validation_opt(self, validation_dataloader, avg_nll_regression, 2, device=device)


def _disable_grads(model, uncertainty_parameters):
    # disable gradient calculations 
    uncertainty_parameters_set = set(uncertainty_parameters)
    enabled_grads = []
    for parameter in model.parameters():
        enabled_grads.append(parameter.requires_grad)
        if parameter in uncertainty_parameters_set:
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
    return enabled_grads


def _enable_grads(model, enabled_grads):
    # restore requires_grad to its original value for all parameters
    for is_enabled, parameter in zip(enabled_grads, model.parameters()):
        parameter.requires_grad_(is_enabled)


def _jacobian(outputs, parameters):
    # output_shape = outputs.shape + (sum([parameter.flatten().shape[0] for parameter in parameters]),)
    # return torch.zeros(output_shape, device=outputs.device, dtype=outputs.dtype)

    out_shape = outputs.shape
    outputs = outputs.flatten()
    jacobian = []
    for output in outputs:
        gradients = torch.autograd.grad(
            output,
            parameters,
            retain_graph=True,
            create_graph=False
        )
        gradients = torch.concatenate([gradient.flatten() for gradient in gradients])
        jacobian.append(gradients)
    jacobian = torch.stack(jacobian)
    jacobian = jacobian.reshape(out_shape + (jacobian.shape[1],))
    return jacobian  # [n_batch, n_out, n_w]


