import torch
import numpy as np
import copy


class DeepEnsemble(torch.nn.Module):
    def __init__(
        self,
        models,
    ):
        super().__init__()
        self.models = models

    def forward(self, x):
        # Apply formulas from the paper
        means = []
        variances = []
        for model in self.models:
            mean, variance = model(x)
            means.append(mean)
            variances.append(variance)
        stacked_means = torch.stack(means)
        stacked_variances = torch.stack(variances)
        final_mean = torch.mean(stacked_means, dim=0)
        final_variance = torch.mean(stacked_variances + stacked_means**2, dim=0) - final_mean**2
        return final_mean, final_variance


class DeepEnsembleLastLayer(torch.nn.Module):
    """A last layer of a neural network that outputs a mean and a variance,
    as described in the Deep Ensembles paper."""

    def __init__(self, n_inputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, 2)

    def forward(self, x):
        x = self.linear(x)
        mean = x[:, 0]
        variance = torch.nn.functional.softplus(x[:, 1])
        return mean.unsqueeze(1), variance.unsqueeze(1)


def nll_loss(mu, sigma_sq, y):
    """Negative log-likelihood loss function for regression
    from the Deep Ensembles paper. The constant term is omitted."""
    return 0.5*(torch.mean(torch.log(sigma_sq) + (y - mu)**2 / sigma_sq) + np.log(2*np.pi))


def train_nll(model, optimizer, nll_loss_fn, train_dataloader, validation_dataloader, n_epochs, device):

    def evaluate_loss(model, dataloader):
        with torch.no_grad():
            mu, sigma_sq, y_actual = [], [], []
            for batch in dataloader:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                mu_batch, sigma_sq_batch = model(X_batch)
                mu.append(mu_batch)
                sigma_sq.append(sigma_sq_batch)
                y_actual.append(y_batch)
            mu = torch.cat(mu)
            sigma_sq = torch.cat(sigma_sq)
            y_actual = torch.cat(y_actual)
            loss = nll_loss_fn(mu, sigma_sq, y_actual).item()
        return loss

    model.eval()
    train_loss = evaluate_loss(model, train_dataloader)
    valid_loss = evaluate_loss(model, validation_dataloader)
    print("Epoch:", 0, " Train Loss:", train_loss, " Valid Loss:", valid_loss)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6)
    best_valid_loss = np.inf
    best_train_loss = np.inf

    # Training loop
    for epoch in range(1, n_epochs+1):

        # Training phase
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mu, sigma_sq = model(X_batch)
            loss = nll_loss_fn(mu, sigma_sq, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluation phase
        model.eval()
        train_loss = evaluate_loss(model, train_dataloader)
        valid_loss = evaluate_loss(model, validation_dataloader)
        print("Epoch:", epoch, " Train Loss:", train_loss, " Valid Loss:", valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_train_loss = train_loss
            best_weights = copy.deepcopy(model.state_dict())

        # Update the learning rate
        scheduler.step(valid_loss)

    model.load_state_dict(best_weights)
