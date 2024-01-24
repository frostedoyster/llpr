# This file contains utilities needed to optimize the llpr
# hypers on a validation set.

import torch
import numpy as np
from scipy.optimize import minimize


def softplus(x):
    return np.log(1.0+np.exp(-np.abs(x))) + max(x, 0.0)


def process_inputs(x):
    x = list(x)
    x = [np.exp(single_x) for single_x in x]
    return x


def validation_opt(model, validation_loader, objective_function, n_parameters, **kwargs):
    # The parameters must be a list of torch tensors.
    # This function optimizes the parameters on the validation set.

    def objective_function_wrapper(x):
        x = process_inputs(x)
        try:
            model.set_hyperparameters(*x)
            objective_function_value = objective_function(model, validation_loader, **kwargs)
        except torch._C._LinAlgError:
            objective_function_value = 1e10
        print(x, objective_function_value)
        return objective_function_value

    # Initial guess
    if n_parameters == 1:
        initial_value = 0.0
        delta = 0.1
        x0 = np.array([initial_value])
        initial_simplex = np.array([
            x0,
            x0 + [delta]   
        ])
    elif n_parameters == 2:
        initial_value_1 = 5.0
        initial_value_2 = -5.0
        delta_1 = 1.0
        delta_2 = 0.05
        initial_simplex = np.array([
            [initial_value_1, initial_value_2],
            [initial_value_1 + delta_1, initial_value_2],
            [initial_value_1, initial_value_2 + delta_2]
        ])
    else:
        raise NotImplementedError()

    result = minimize(objective_function_wrapper, [0.0 for _ in range(n_parameters)], method="Nelder-Mead", options={"initial_simplex": initial_simplex, "maxiter": 50})
    print("Optimal parameters:", process_inputs(result.x))
    print("Optimal value:", result.fun)

    model.set_hyperparameters(*(process_inputs(result.x)))


