import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


actual_variances = np.load("outputs/figures/actual_variances.npy")
estimated_variances = np.load("outputs/figures/estimated_variances.npy")

n_per_bin = 100

sorting = np.argsort(estimated_variances)
estimated_variances = estimated_variances[sorting]
actual_variances = actual_variances[sorting]

bins = np.arange(0, len(estimated_variances), n_per_bin)
estimated_variances_avg = np.array([np.mean(estimated_variances[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])
actual_variances_avg = np.array([np.mean(actual_variances[bins[i] : bins[i + 1]]) for i in range(len(bins) - 1)])

# Remove the last bin if it is not full
if len(estimated_variances) % n_per_bin != 0:
    estimated_variances_avg = estimated_variances_avg[:-1]
    actual_variances_avg = actual_variances_avg[:-1]

min_value = min(np.min(estimated_variances_avg), np.min(actual_variances_avg))
max_value = max(np.max(estimated_variances_avg), np.max(actual_variances_avg))

plt.plot(estimated_variances_avg, actual_variances_avg, ".", markersize=20)
plt.plot([min_value, max_value], [min_value, max_value], "k", label="y=x")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Estimated variance")
plt.ylabel("Mean squared error")
plt.xticks([3, 4, 6, 10], [r"$3 \times 10^{0}$", "", r"$6 \times 10^{0}$", r"$10^{1}$"])
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/aus_weather_bin.pdf")
