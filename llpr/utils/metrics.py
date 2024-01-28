import numpy as np
import torch
import matplotlib.pyplot as plt

from .to_device import to_device


def avg_nll_regression(model, dataloader, device=None, train_mean_y=0.0, train_std_y=1.0):
    # calculates a NLL on a dataset

    total_nll = 0.0
    total_datapoints = 0
    for batch in dataloader:
        x = batch[:-1]
        if len(x) == 1: x = x[0]
        y = batch[-1]
        x, y = to_device(device, x, y)
        y = y * train_std_y + train_mean_y
        predictions, estimated_variances = model(x)
        predictions = predictions * train_std_y + train_mean_y
        estimated_variances = estimated_variances * train_std_y**2
        total_datapoints += len(x)
        total_nll += (
            (y-predictions)**2 / estimated_variances + torch.log(estimated_variances) + np.log(2*np.pi)
        ).sum().item() * 0.5

    return total_nll / total_datapoints


def sum_squared_log(model, dataloader, device=None):

    actual_errors = []
    predicted_errors = []
    for batch in dataloader:
        x = batch[:-1]
        if len(x) == 1: x = x[0]
        y = batch[-1]
        x, y = to_device(device, x, y)
        predictions, estimated_variances = model(x)
        actual_errors.append((y-predictions)**2)
        predicted_errors.append(estimated_variances)
    actual_errors = torch.cat(actual_errors)
    predicted_errors = torch.cat(predicted_errors)

    sort_indices = torch.argsort(predicted_errors)
    actual_errors = actual_errors[sort_indices]
    predicted_errors = predicted_errors[sort_indices]

    n_samples_per_bin = 100
    n_samples = len(actual_errors)

    actual_error_bins = []
    predicted_error_bins = []

    # skip the last bin, which will most likely be incomplete
    for i_bin in range(n_samples // n_samples_per_bin - 1):
        actual_error_bins.append(
            actual_errors[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
        )
        predicted_error_bins.append(
            predicted_errors[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
        )
    
    actual_error_bins = torch.stack(actual_error_bins)
    predicted_error_bins = torch.stack(predicted_error_bins)

    # calculate means:
    actual_error_means = actual_error_bins.mean(dim=1)
    predicted_error_means = predicted_error_bins.mean(dim=1)

    # calculate squared log errors:
    squared_log_errors = (
        torch.log(actual_error_means / predicted_error_means)**2
    )

    # calculate the sum of squared log errors:
    sum_squared_log = squared_log_errors.sum().item()

    return sum_squared_log


def rmse(model, dataloader, device=None, train_mean_y=0.0, train_std_y=1.0):
    # calculates a RMSE on a dataset

    total_se = 0.0
    total_datapoints = 0
    for batch in dataloader:
        x = batch[:-1]
        if len(x) == 1: x = x[0]
        y = batch[-1]
        x, y = to_device(device, x, y)
        y = y * train_std_y + train_mean_y
        predictions, estimated_variances = model(x)
        predictions = predictions * train_std_y + train_mean_y
        estimated_variances = estimated_variances * train_std_y**2
        total_datapoints += len(x)
        total_se += (
            (y-predictions)**2
        ).sum().item()

    return np.sqrt(total_se / total_datapoints)


def mae(model, dataloader, device=None):
    # calculates a MAE on a dataset

    total_ae = 0.0
    total_datapoints = 0
    for batch in dataloader:
        x = batch[:-1]
        if len(x) == 1: x = x[0]
        y = batch[-1]
        x, y = to_device(device, x, y)
        predictions, estimated_variances = model(x)
        total_datapoints += len(x)
        total_ae += (
            torch.abs(y-predictions)
        ).sum().item()

    return total_ae / total_datapoints


def regression_uncertainty_parity_plot(model, dataloader, device, plot_path, n_samples_per_bin=None, train_mean_y=0.0, train_std_y=1.0):

    estimated_errors = []
    actual_errors = []
    for batch in dataloader:
        input, result = batch 
        input, result = input.to(device), result.to(device)
        result = result * train_std_y + train_mean_y
        prediction, uncertainty = model(input)
        prediction = prediction * train_std_y + train_mean_y
        uncertainty = uncertainty * train_std_y**2
        estimated_errors.append(uncertainty)
        actual_errors.append(
            (prediction-result)**2
        )
    actual_errors = torch.concatenate(actual_errors).detach().squeeze(1).cpu().numpy()
    estimated_errors = torch.concatenate(estimated_errors).detach().squeeze(1).cpu().numpy()

    n_sigma = 1  # hardcoded: pretty much the only sound choice
    if n_samples_per_bin is None:
        n_samples_per_bin = int(np.sqrt(len(estimated_errors)))  # just a default choice

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

    min_avg = min(np.min(estimated_averages), np.min(actual_averages))
    max_avg = max(np.max(estimated_averages), np.max(actual_averages))
    plt.plot(actual_averages, estimated_averages, ".")
    plt.plot([min_avg, max_avg], [min_avg, max_avg], label="y=x")
    plt.fill_between(
        [min_avg, max_avg],
        [min_avg+n_sigma*np.sqrt(2.0/n_samples_per_bin)*min_avg, max_avg+n_sigma*np.sqrt(2.0/n_samples_per_bin)*max_avg],
        [min_avg-n_sigma*np.sqrt(2.0/n_samples_per_bin)*min_avg, max_avg-n_sigma*np.sqrt(2.0/n_samples_per_bin)*max_avg],
        alpha=0.3,
        color="tab:orange"
    )
    plt.ylabel("Predicted errors")
    plt.xlabel("Actual errors")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(plot_path)
