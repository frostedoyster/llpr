import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

import llpr

torch.set_default_dtype(torch.float64)



def test_create_then_script():
    """Test that the uncertainty model is torch-scriptable and
    that the results are consistent."""

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        torch.nn.LayerNorm(32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 32),
        torch.nn.LayerNorm(32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 1, bias=False)
    )

    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, shuffle=True)
    
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64).reshape(-1, 1)

    batch_size = 10
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)

    loss_fn = torch.nn.MSELoss(reduction="sum")
    model_llpr = llpr.UncertaintyModel(model, model[-1], train_dataloader)
    # model_pr = llpr.pr.UncertaintyModel(model, [model[-1]], train_dataloader, loss_fn)
    # model_pr_loss_agnostic = llpr.pr.UncertaintyModel(model, [model[-1]], train_dataloader)

    model_llpr.set_hyperparameters(1.0, 1.0)
    # model_pr.set_hyperparameters(2.0, np.sqrt(2.0))
    # model_pr_loss_agnostic.set_hyperparameters(1.0, 1.0)

    model_llpr_scripted = torch.jit.script(model_llpr)
    # model_pr_scripted = torch.jit.script(model_pr)
    # model_pr_loss_agnostic_scripted = torch.jit.script(model_pr_loss_agnostic)

    x, _ = next(iter(test_dataloader))

    predictions_llpr, uncertainties_llpr = model_llpr(x)
    # predictions_pr, uncertainties_pr = model_pr(x)
    # predictions_pr_loss_agnostic, uncertainties_pr_loss_agnostic = model_pr_loss_agnostic(x)

    predictions_llpr_scripted, uncertainties_llpr_scripted = model_llpr_scripted(x)
    # predictions_pr_scripted, uncertainties_pr_scripted = model_pr_scripted(x)
    # predictions_pr_loss_agnostic_scripted, uncertainties_pr_loss_agnostic_scripted = model_pr_loss_agnostic_scripted(x)

    assert torch.allclose(predictions_llpr, predictions_llpr_scripted)
    assert torch.allclose(uncertainties_llpr, uncertainties_llpr_scripted)
    # assert torch.allclose(predictions_pr, predictions_pr_scripted)
    # assert torch.allclose(uncertainties_pr, uncertainties_pr_scripted)
    # assert torch.allclose(predictions_pr_loss_agnostic, predictions_pr_loss_agnostic_scripted)
    # assert torch.allclose(uncertainties_pr_loss_agnostic, uncertainties_pr_loss_agnostic_scripted)
