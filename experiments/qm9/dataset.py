import torch
import ase.io
import numpy as np

import rascaline.torch


def get_dataset():

    structures = ase.io.read("data/qm9.xyz", ":")
    np.random.shuffle(structures)
    n_train = 10000
    n_valid = 10000
    # the rest is for testing

    train_structures = structures[:n_train]
    valid_structures = structures[n_train:n_train+n_valid]
    test_structures = structures[n_train+n_valid:]
    target_key = "U0"

    composition_calculator = rascaline.torch.AtomicComposition(per_structure=True)
    train_composition = composition_calculator.compute(rascaline.torch.systems_to_torch(train_structures))
    valid_composition = composition_calculator.compute(rascaline.torch.systems_to_torch(valid_structures))
    test_composition = composition_calculator.compute(rascaline.torch.systems_to_torch(test_structures))

    train_composition = train_composition.keys_to_properties("species_center").block().values
    valid_composition = valid_composition.keys_to_properties("species_center").block().values
    test_composition = test_composition.keys_to_properties("species_center").block().values

    assert train_composition.shape == (n_train, 5)
    assert valid_composition.shape == (n_valid, 5)
    assert test_composition.shape == (len(test_structures), 5)

    train_targets = torch.tensor([structure.info[target_key] for structure in train_structures])
    valid_targets = torch.tensor([structure.info[target_key] for structure in valid_structures])
    test_targets = torch.tensor([structure.info[target_key] for structure in test_structures])

    composition_coefficients = torch.linalg.solve(train_composition.T @ train_composition, train_composition.T @ train_targets)

    train_targets = train_targets - train_composition @ composition_coefficients
    valid_targets = valid_targets - valid_composition @ composition_coefficients
    test_targets = test_targets - test_composition @ composition_coefficients

    for i_structure in range(len(train_structures)):
        train_structures[i_structure].info[target_key] = train_targets[i_structure].item()

    for i_structure in range(len(valid_structures)):
        valid_structures[i_structure].info[target_key] = valid_targets[i_structure].item()

    for i_structure in range(len(test_structures)):
        test_structures[i_structure].info[target_key] = test_targets[i_structure].item()

    train_dataset = [(structure, structure.info["U0"]) for structure in train_structures]
    valid_dataset = [(structure, structure.info["U0"]) for structure in valid_structures]
    test_dataset = [(structure, structure.info["U0"]) for structure in test_structures]

    return train_dataset, valid_dataset, test_dataset
