import numpy as np


with open(f"results.out", "w") as f:
    f.write("")

for dataset in ["concrete", "energy", "kin8nm", "naval", "power", "protein", "wine", "yacht", "year"]:
    rmses = []
    nlls = []
    n_seeds = 20
    if dataset == "protein": n_seeds = 5
    if dataset == "year": n_seeds = 1
    for seed in range(n_seeds):
        with open(f"outputs/llpr_{dataset}_{seed}.out", "r") as f:
            lines = f.readlines()
            rmse = float(lines[-1].split(" ")[-1])
            nll = float(lines[-2].split(" ")[-1])
            rmses.append(rmse)
            nlls.append(nll)
    rmse = np.array(rmses)
    nll = np.array(nlls)
    with open(f"results.out", "a") as f:
        f.write(f"{dataset}\n")
        f.write(f"RMSE: {rmse.mean()} +/- {rmse.std()/np.sqrt(5.0)}\n")
        f.write(f"NLL:  {nll.mean()} +/- {nll.std()/np.sqrt(5.0)}\n")
