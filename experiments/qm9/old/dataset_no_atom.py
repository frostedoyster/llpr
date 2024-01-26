import torch
import ase.io
import numpy as np


def get_dataset():

    structures = ase.io.read("data/qm9.xyz", ":")
    np.random.shuffle(structures)
    n_train = 50000
    n_valid = 5000
    # the rest is for testing

    train_structures = structures[:n_train]
    valid_structures = structures[n_train:n_train+n_valid]
    test_structures = structures[n_train+n_valid:]

    train_dataset = [(structure, structure.info["U0"]) for structure in train_structures]
    valid_dataset = [(structure, structure.info["U0"]) for structure in valid_structures]
    test_dataset = [(structure, structure.info["U0"]) for structure in test_structures]

    return train_dataset, valid_dataset, test_dataset
