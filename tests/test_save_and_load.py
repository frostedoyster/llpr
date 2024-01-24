import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

import llpr

torch.set_default_dtype(torch.float64)


def test_save_load():
    """Test that the uncertainty model is saved and loaded correctly."""

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
    model_pr = llpr.pr.UncertaintyModel(model, [model[-1]], train_dataloader, loss_fn)
    model_pr_loss_agnostic = llpr.pr.UncertaintyModel(model, [model[-1]], train_dataloader)

    model_llpr.set_hyperparameters(1.0, 1.0)
    model_pr.set_hyperparameters(2.0, np.sqrt(2.0))
    model_pr_loss_agnostic.set_hyperparameters(1.0, 1.0)

    # save models
    torch.save(model_llpr, "model_llpr.pt")
    torch.save(model_pr, "model_pr.pt")
    torch.save(model_pr_loss_agnostic, "model_pr_loss_agnostic.pt")

    # load models
    model_llpr_loaded = torch.load("model_llpr.pt")
    model_pr_loaded = torch.load("model_pr.pt")
    model_pr_loss_agnostic_loaded = torch.load("model_pr_loss_agnostic.pt")

    x, _ = next(iter(test_dataloader))
    predictions_llpr, uncertainties_llpr = model_llpr(x)
    predictions_pr, uncertainties_pr = model_pr(x)
    predictions_pr_loss_agnostic, uncertainties_pr_loss_agnostic = model_pr_loss_agnostic(x)

    predictions_llpr_loaded, uncertainties_llpr_loaded = model_llpr_loaded(x)
    predictions_pr_loaded, uncertainties_pr_loaded = model_pr_loaded(x)
    predictions_pr_loss_agnostic_loaded, uncertainties_pr_loss_agnostic_loaded = model_pr_loss_agnostic_loaded(x)

    assert torch.allclose(predictions_llpr, predictions_llpr_loaded)
    assert torch.allclose(predictions_pr, predictions_pr_loaded)
    assert torch.allclose(predictions_pr_loss_agnostic, predictions_pr_loss_agnostic_loaded)
    assert torch.allclose(uncertainties_llpr, uncertainties_llpr_loaded)
    assert torch.allclose(uncertainties_pr, uncertainties_pr_loaded)
    assert torch.allclose(uncertainties_pr_loss_agnostic, uncertainties_pr_loss_agnostic_loaded)


def test_save_load_state_dict():
    """Test that the state dict of the uncertainty model is saved and loaded correctly."""

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
    model_pr = llpr.pr.UncertaintyModel(model, [model[-1]], train_dataloader, loss_fn)
    model_pr_loss_agnostic = llpr.pr.UncertaintyModel(model, [model[-1]], train_dataloader)

    model_llpr.set_hyperparameters(1.0, 1.0)
    model_pr.set_hyperparameters(2.0, np.sqrt(2.0))
    model_pr_loss_agnostic.set_hyperparameters(1.0, 1.0)

    # save state dicts
    torch.save(model_llpr.state_dict(), "model_llpr.pt")
    torch.save(model_pr.state_dict(), "model_pr.pt")
    torch.save(model_pr_loss_agnostic.state_dict(), "model_pr_loss_agnostic.pt")

    # load state dicts
    model_llpr_dict_loaded = torch.load("model_llpr.pt")
    model_pr_dict_loaded = torch.load("model_pr.pt")
    model_pr_loss_agnostic_dict_loaded = torch.load("model_pr_loss_agnostic.pt")

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        torch.nn.LayerNorm(32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 32),
        torch.nn.LayerNorm(32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 1, bias=False)
    )

    model_llpr_loaded = llpr.UncertaintyModel(model, model[-1])
    model_pr_loaded = llpr.pr.UncertaintyModel(model, [model[-1]])
    model_pr_loss_agnostic_loaded = llpr.pr.UncertaintyModel(model, [model[-1]])

    model_llpr_loaded.load_state_dict(model_llpr_dict_loaded)
    model_pr_loaded.load_state_dict(model_pr_dict_loaded)
    model_pr_loss_agnostic_loaded.load_state_dict(model_pr_loss_agnostic_dict_loaded)

    x, _ = next(iter(test_dataloader))
    predictions_llpr, uncertainties_llpr = model_llpr(x)
    predictions_pr, uncertainties_pr = model_pr(x)
    predictions_pr_loss_agnostic, uncertainties_pr_loss_agnostic = model_pr_loss_agnostic(x)

    predictions_llpr_loaded, uncertainties_llpr_loaded = model_llpr_loaded(x)
    predictions_pr_loaded, uncertainties_pr_loaded = model_pr_loaded(x)
    predictions_pr_loss_agnostic_loaded, uncertainties_pr_loss_agnostic_loaded = model_pr_loss_agnostic_loaded(x)

    assert torch.allclose(predictions_llpr, predictions_llpr_loaded)
    assert torch.allclose(predictions_pr, predictions_pr_loaded)
    assert torch.allclose(predictions_pr_loss_agnostic, predictions_pr_loss_agnostic_loaded)
    assert torch.allclose(uncertainties_llpr, uncertainties_llpr_loaded)
    assert torch.allclose(uncertainties_pr, uncertainties_pr_loaded)
    assert torch.allclose(uncertainties_pr_loss_agnostic, uncertainties_pr_loss_agnostic_loaded)
