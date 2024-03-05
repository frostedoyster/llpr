


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from scipy.optimize import root_scalar
from scipy.stats import norm


def pdf(x, sigma):
    return x * np.exp(-x**2/(2*sigma**2)) * 1.0/(sigma*np.sqrt(2*np.pi))

def find_where_pdf_is_c(c, sigma):
    # Finds the two values of x where the pdf is equal to c
    mode_value = pdf(sigma, sigma)
    if c > mode_value:
        raise ValueError("c must be less than mode_value")
    where_below_mode = root_scalar(lambda x: pdf(x, sigma) - c, bracket=[0, sigma]).root
    where_above_mode = root_scalar(lambda x: pdf(x, sigma) - c, bracket=[sigma, 100]).root
    return where_below_mode, where_above_mode

def pdf_integral(sigma, c):
    # Calculates the integral (analytical) of the pdf from x1 to x2,
    # where x1 and x2 are the two values of x where the pdf is equal to c
    x1, x2 = find_where_pdf_is_c(c, sigma)
    return np.exp(-x1**2/(2*sigma**2)) - np.exp(-x2**2/(2*sigma**2))

def find_fraction(sigma, fraction):
    # Finds the value of c where the integral of the pdf from x1 to x2 is equal to fraction,
    # where x1 and x2 are the two values of x where the pdf is equal to c
    mode_value = pdf(sigma, sigma)
    return root_scalar(lambda x: pdf_integral(sigma, x) - fraction, x0=mode_value-0.01, x1=mode_value-0.02).root

estimated_variances_valid = np.load("outputs/figures/estimated_variances_valid.npy")
actual_variances_valid = np.load("outputs/figures/actual_variances_valid.npy")
estimated_variances_test = np.load("outputs/figures/estimated_variances_test.npy")
actual_variances_test = np.load("outputs/figures/actual_variances_test.npy")
estimated_variances_ood = np.load("outputs/figures/estimated_variances_ood.npy")
actual_variances_ood = np.load("outputs/figures/actual_variances_ood.npy")

# we transform these "squared errors" and "predicted variances" into
# "absolute errors" and "predicted standard deviations"

estimated_variances_valid = np.sqrt(estimated_variances_valid)
actual_variances_valid = np.sqrt(actual_variances_valid)
estimated_variances_test = np.sqrt(estimated_variances_test)
actual_variances_test = np.sqrt(actual_variances_test)
estimated_variances_ood = np.sqrt(estimated_variances_ood)
actual_variances_ood = np.sqrt(actual_variances_ood)

min_value_estimated = min(np.min(estimated_variances_test), np.min(estimated_variances_ood))
min_value_actual = min(np.min(actual_variances_test), np.min(actual_variances_ood))
min_value = max(min_value_estimated, min_value_actual)

max_value_estimated = max(np.max(estimated_variances_test), np.max(estimated_variances_ood))
max_value_actual = max(np.max(actual_variances_test), np.max(actual_variances_ood))
max_value = min(max_value_estimated, max_value_actual)

desired_fractions = [
    norm.cdf(1, 0.0, 1.0) - norm.cdf(-1, 0.0, 1.0),  # 1 sigma
    norm.cdf(2, 0.0, 1.0) - norm.cdf(-2, 0.0, 1.0),  # 2 sigma
    norm.cdf(3, 0.0, 1.0) - norm.cdf(-3, 0.0, 1.0),  # 3 sigma
]
sigmas = [min_value, max_value]
lower_bounds = []
upper_bounds = []
for desired_fraction in desired_fractions:
    lower_bounds.append([])
    upper_bounds.append([])
    for sigma in sigmas:
        isoline_value = find_fraction(sigma, desired_fraction)
        x1, x2 = find_where_pdf_is_c(isoline_value, sigma)
        lower_bounds[-1].append(x1)
        upper_bounds[-1].append(x2)
    lower_bounds[-1] = np.array(lower_bounds[-1])
    upper_bounds[-1] = np.array(upper_bounds[-1])

plt.plot([min_value, max_value], [min_value, max_value], "k", label="y=x")
for i, desired_fraction in enumerate(desired_fractions):
    plt.plot(sigmas, lower_bounds[i], "k", alpha=1.0-(i+1)*0.2, linewidth=1.0-(i+1)*0.2)
    plt.plot(sigmas, upper_bounds[i], "k", alpha=1.0-(i+1)*0.2, linewidth=1.0-(i+1)*0.2)

print(estimated_variances_valid.shape)
print(actual_variances_valid.shape)
print(estimated_variances_test.shape)
print(actual_variances_test.shape)
print(estimated_variances_ood.shape)
print(actual_variances_ood.shape)
plt.scatter(estimated_variances_test, actual_variances_test, s=10.0, edgecolors='none', alpha=0.5, label="In domain")
plt.scatter(estimated_variances_ood, actual_variances_ood, s=10.0, edgecolors='none', alpha=0.5, label="Out of domain")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Estimated error")
plt.ylabel("Absolute error")
plt.xticks([0.3, 1, 3], [r"$3 \times 10^{-1}$", r"$10^{0}$", r"$3 \times 10^{0}$"])
plt.legend()
plt.tight_layout()
plt.savefig(f"outputs/figures/ood_cali_no_bin.pdf")
