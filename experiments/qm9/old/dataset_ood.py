import torch
import ase.io
import numpy as np

import rascaline.torch


def get_dataset():

    structures = ase.io.read("data/qm9.xyz", ":")
    number_of_oxygen_atoms = []
    for structure in structures:
        number_of_oxygen_atoms.append(
            len(np.where(structure.numbers == 8)[0])
        )


    mask_n_oxygens = []
    for i in range(6):
        mask_n_oxygens.append(
            np.array(number_of_oxygen_atoms) == i
        )

    # Select structures with 0, 1, 4, 5 atoms using the masks
    structures_0_oxygens = []
    structures_1_oxygens = []
    structures_4_oxygens = []
    structures_5_oxygens = []
    for i_structure in range(len(structures)):
        if mask_n_oxygens[0][i_structure]:
            structures_0_oxygens.append(structures[i_structure])
        elif mask_n_oxygens[1][i_structure]:
            structures_1_oxygens.append(structures[i_structure])
        elif mask_n_oxygens[4][i_structure]:
            structures_4_oxygens.append(structures[i_structure])
        elif mask_n_oxygens[5][i_structure]:
            structures_5_oxygens.append(structures[i_structure])
        else:
            pass

    train_valid_test_structures = structures_0_oxygens + structures_1_oxygens
    ood_structures = structures_4_oxygens + structures_5_oxygens

    # shuffle the structures
    np.random.shuffle(train_valid_test_structures)
    np.random.shuffle(ood_structures)

    # split into train, valid, test
    n_train = 50000
    n_valid = 5000

    train_structures = train_valid_test_structures[:n_train]
    valid_structures = train_valid_test_structures[n_train:n_train+n_valid]
    test_structures = train_valid_test_structures[n_train+n_valid:]

    target_key = "U0"

    composition_calculator = rascaline.torch.AtomicComposition(per_structure=True)
    train_composition = composition_calculator.compute(rascaline.torch.systems_to_torch(train_structures))
    valid_composition = composition_calculator.compute(rascaline.torch.systems_to_torch(valid_structures))
    test_composition = composition_calculator.compute(rascaline.torch.systems_to_torch(test_structures))
    ood_composition = composition_calculator.compute(rascaline.torch.systems_to_torch(ood_structures))

    train_composition = train_composition.keys_to_properties("species_center").block().values
    valid_composition = valid_composition.keys_to_properties("species_center").block().values
    test_composition = test_composition.keys_to_properties("species_center").block().values
    ood_composition = ood_composition.keys_to_properties("species_center").block().values

    assert train_composition.shape == (n_train, 5)
    assert valid_composition.shape == (n_valid, 5)
    assert test_composition.shape == (len(test_structures), 5)
    assert ood_composition.shape == (len(ood_structures), 5)

    train_targets = torch.tensor([structure.info[target_key] for structure in train_structures])
    valid_targets = torch.tensor([structure.info[target_key] for structure in valid_structures])
    test_targets = torch.tensor([structure.info[target_key] for structure in test_structures])
    ood_targets = torch.tensor([structure.info[target_key] for structure in ood_structures])

    composition_coefficients = torch.linalg.solve(train_composition.T @ train_composition, train_composition.T @ train_targets)

    train_targets = train_targets - train_composition @ composition_coefficients
    valid_targets = valid_targets - valid_composition @ composition_coefficients
    test_targets = test_targets - test_composition @ composition_coefficients
    ood_targets = ood_targets - ood_composition @ composition_coefficients

    for i_structure in range(len(train_structures)):
        train_structures[i_structure].info[target_key] = train_targets[i_structure].item()

    for i_structure in range(len(valid_structures)):
        valid_structures[i_structure].info[target_key] = valid_targets[i_structure].item()

    for i_structure in range(len(test_structures)):
        test_structures[i_structure].info[target_key] = test_targets[i_structure].item()

    for i_structure in range(len(ood_structures)):
        ood_structures[i_structure].info[target_key] = ood_targets[i_structure].item()

    train_dataset = [(structure, structure.info["U0"]) for structure in train_structures]
    valid_dataset = [(structure, structure.info["U0"]) for structure in valid_structures]
    test_dataset = [(structure, structure.info["U0"]) for structure in test_structures]
    ood_dataset = [(structure, structure.info["U0"]) for structure in ood_structures]

    return train_dataset, valid_dataset, test_dataset, ood_dataset


if __name__ == "__main__":
    train_structures, valid_structures, test_structures, ood_structures = get_dataset()
    print(len(train_structures))
    print(len(valid_structures))
    print(len(test_structures))
    print(len(ood_structures))
