# This file contains utilities needed to optimize the llpr
# hypers on a validation set.

import torch
import numpy as np
from scipy.optimize import brute


def process_inputs(x):
    x = list(x)
    x = [10**single_x for single_x in x]
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

    assert n_parameters == 2

    result = brute(objective_function_wrapper, ranges=[slice(-5, 5, 1.0), slice(-5, 5, 1.0)])
    print("Optimal parameters:", process_inputs(result))
    # print("Optimal value:", result.fval)

    # warn if we hit the edge of the parameter space
    if result[0] == -5 or result[0] == 5 or result[1] == -5 or result[1] == 5:
        print("WARNING: Hit the edge of the parameter space")

    model.set_hyperparameters(*(process_inputs(result)))


