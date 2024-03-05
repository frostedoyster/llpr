import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


actual_variances_valid = np.load("outputs/figures/actual_variances_valid.npy")
estimated_variances_valid = np.load("outputs/figures/estimated_variances_valid.npy")
actual_variances_test = np.load("outputs/figures/actual_variances_test.npy")
estimated_variances_test = np.load("outputs/figures/estimated_variances_test.npy")
actual_variances_ood = np.load("outputs/figures/actual_variances_ood.npy")
estimated_variances_ood = np.load("outputs/figures/estimated_variances_ood.npy")

n_per_bin = 100

sorting = np.argsort(estimated_variances_valid)
estimated_variances_valid = estimated_variances_valid[sorting]
actual_variances_valid = actual_variances_valid[sorting]

bins = np.arange(0, len(estimated_variances_valid), n_per_bin)
estimated_variances_valid_avg = np.array([np.mean(estimated_variances_valid[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_valid_avg = np.array([np.mean(actual_variances_valid[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Exclude the last bin, which will most likely be incomplete:
estimated_variances_valid_avg = estimated_variances_valid_avg[:-1]
actual_variances_valid_avg = actual_variances_valid_avg[:-1]

sorting = np.argsort(estimated_variances_test)
estimated_variances_test = estimated_variances_test[sorting]
actual_variances_test = actual_variances_test[sorting]

bins = np.arange(0, len(estimated_variances_test), n_per_bin)
estimated_variances_test_avg = np.array([np.mean(estimated_variances_test[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_test_avg = np.array([np.mean(actual_variances_test[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Exclude the last bin, which will most likely be incomplete:
estimated_variances_test_avg = estimated_variances_test_avg[:-1]
actual_variances_test_avg = actual_variances_test_avg[:-1]

sorting = np.argsort(estimated_variances_ood)
estimated_variances_ood = estimated_variances_ood[sorting]
actual_variances_ood = actual_variances_ood[sorting]

bins = np.arange(0, len(estimated_variances_ood), n_per_bin)
estimated_variances_ood_avg = np.array([np.mean(estimated_variances_ood[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_ood_avg = np.array([np.mean(actual_variances_ood[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Exclude the last bin, which will most likely be incomplete:
estimated_variances_ood_avg = estimated_variances_ood_avg[:-1]
actual_variances_ood_avg = actual_variances_ood_avg[:-1]

min_value = min(np.min(estimated_variances_test_avg), np.min(actual_variances_test_avg), np.min(estimated_variances_ood_avg), np.min(actual_variances_ood_avg))
max_value = max(np.max(estimated_variances_test_avg), np.max(actual_variances_test_avg), np.max(estimated_variances_ood_avg), np.max(actual_variances_ood_avg))

plt.plot([min_value, max_value], [min_value, max_value], "k", label="y=x")
print(estimated_variances_valid_avg.shape)
print(actual_variances_valid_avg.shape)
print(estimated_variances_test_avg.shape)
print(actual_variances_test_avg.shape)
print(estimated_variances_ood_avg.shape)
print(actual_variances_ood_avg.shape)
# plt.plot(estimated_variances_valid_avg, actual_variances_valid_avg, "o")
plt.plot(estimated_variances_test_avg, actual_variances_test_avg, ".", markersize=20, label="In domain")
plt.plot(estimated_variances_ood_avg, actual_variances_ood_avg, ".", markersize=20, label="Out of domain")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Estimated variance")
plt.ylabel("Mean squared error")
plt.legend()
plt.tight_layout()
plt.savefig(f"outputs/figures/ood_cali_bin.pdf")
