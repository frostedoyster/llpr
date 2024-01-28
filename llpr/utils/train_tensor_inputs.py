import torch
import numpy as np
import copy

from .to_device import to_device


def train_model(model, optimizer, loss_fn, train_dataloader, validation_dataloader, n_epochs, device):

    def evaluate_loss(model, dataloader):
        with torch.no_grad():
            y_pred, y_actual = [], []
            for batch in dataloader:
                X_batch = batch[:-1]
                if len(X_batch) == 1: X_batch = X_batch[0]
                y_batch = batch[-1]
                X_batch, y_batch = to_device(device, X_batch, y_batch)
                y_pred_batch = model(X_batch)
                y_pred.append(y_pred_batch)
                y_actual.append(y_batch)
            y_pred = torch.cat(y_pred)
            y_actual = torch.cat(y_actual)
            loss = loss_fn(y_pred, y_actual).item()
        return loss

    model.eval()
    train_loss = evaluate_loss(model, train_dataloader)
    valid_loss = evaluate_loss(model, validation_dataloader)
    print("Epoch:", 0, " Train Loss:", train_loss, " Valid Loss:", valid_loss)

    best_valid_loss = np.inf
    best_train_loss = np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, min_lr=1e-6, verbose=True)

    # Training loop
    for epoch in range(1, n_epochs+1):

        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            X_batch = batch[:-1]
            if len(X_batch) == 1: X_batch = X_batch[0]
            y_batch = batch[-1]
            X_batch, y_batch = to_device(device, X_batch, y_batch)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
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
