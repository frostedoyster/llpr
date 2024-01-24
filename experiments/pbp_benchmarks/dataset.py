import numpy as np
import torch


def get_dataset(name):
  
    if name == "cali":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        features, targets = data.data, data.target
        features = torch.tensor(features, dtype=torch.get_default_dtype())
        targets = torch.tensor(targets, dtype=torch.get_default_dtype())
        number_of_data = targets.shape[0]
        permutation = torch.tensor(np.random.permutation(number_of_data), dtype=torch.int64)
        features = features[permutation]
        targets = targets[permutation]
        n_train = 8 * number_of_data // 10
        n_valid = number_of_data // 10
        n_test = number_of_data - n_train
        X_train = features[:n_train]
        X_valid = features[n_train:n_train+n_valid]
        X_test = features[n_train+n_valid:]
        y_train = targets[:n_train].reshape(-1, 1)
        y_valid = targets[n_train:n_train+n_valid].reshape(-1, 1)
        y_test = targets[n_train+n_valid:].reshape(-1, 1)
    elif name == "year":
        import pandas as pd
        file_path = 'data/year/data.csv'
        df = pd.read_csv(file_path)
        features = torch.tensor(df.iloc[:, 1:].values, dtype=torch.get_default_dtype())
        targets = torch.tensor(df.iloc[:, 0].values, dtype=torch.get_default_dtype())
        number_of_data = targets.shape[0]
        train_validation_features = features[:463715]
        test_features = features[463715:]
        train_validation_targets = targets[:463715]
        test_targets = targets[463715:]
        permutation = torch.tensor(np.random.permutation(463715), dtype=torch.int64)
        train_validation_features = train_validation_features[permutation]
        train_validation_targets = train_validation_targets[permutation]
        n_train = 463715 * 9 // 10
        n_valid = 463715 // 10
        X_train = train_validation_features[:n_train]
        X_valid = train_validation_features[n_train:n_train+n_valid]
        X_test = test_features
        y_train = train_validation_targets[:n_train].reshape(-1, 1)
        y_valid = train_validation_targets[n_train:n_train+n_valid].reshape(-1, 1)
        y_test = test_targets.reshape(-1, 1)
    else:
        file_path = 'data/' + name + '/data.txt'
        data = np.loadtxt(file_path)
        features = torch.tensor(data[:, :-1], dtype=torch.get_default_dtype())
        targets = torch.tensor(data[:, -1], dtype=torch.get_default_dtype())
        number_of_data = targets.shape[0]
        permutation = torch.tensor(np.random.permutation(number_of_data), dtype=torch.int64)
        features = features[permutation]
        targets = targets[permutation]
        n_train = 8 * number_of_data // 10
        n_valid = number_of_data // 10
        n_test = number_of_data - n_train
        X_train = features[:n_train]
        X_valid = features[n_train:n_train+n_valid]
        X_test = features[n_train+n_valid:]
        y_train = targets[:n_train].reshape(-1, 1)
        y_valid = targets[n_train:n_train+n_valid].reshape(-1, 1)
        y_test = targets[n_train+n_valid:].reshape(-1, 1)
    
    mean_X = torch.mean(X_train, dim=0)
    std_X = torch.std(X_train, dim=0, correction=0)
    std_X[std_X == 0.0] = 1.0  # avoid division by zero
    X_train = (X_train - mean_X) / std_X
    X_valid = (X_valid - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X

    mean_y = torch.mean(y_train)
    std_y = torch.std(y_train, correction=0)
    y_train = (y_train - mean_y) / std_y
    y_valid = (y_valid - mean_y) / std_y
    y_test = (y_test - mean_y) / std_y

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset, mean_y, std_y
