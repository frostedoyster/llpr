import torch

import metatensor.torch
import rascaline.torch


class FeatureCalculator(torch.nn.Module):
    def __init__(self, all_species, l_max, n_max, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        hypers = {
            "cutoff": 5.0,
            "max_radial": n_max,
            "max_angular": l_max,
            "atomic_gaussian_width": 0.3,
            "center_atom_weight": 0.0,
            "radial_basis": {
                "Gto": {},
            },
            "cutoff_function": {
                "ShiftedCosine": {"width": 1.0},
            },
            "radial_scaling": {
                "Willatt2018": {
                    "rate": 1.0,
                    "scale": 2.0,
                    "exponent": 7.0
                }
            }
        }
        self.feature_calculator = rascaline.torch.SoapPowerSpectrum(**hypers)
        self.neighbor_labels_1 = metatensor.torch.Labels(
            names=["species_neighbor_1"],
            values=torch.tensor(all_species).reshape(-1, 1),
        )
        self.neighbor_labels_2 = metatensor.torch.Labels(
            names=["species_neighbor_2"],
            values=torch.tensor(all_species).reshape(-1, 1),
        )
        self.all_species = all_species
        self.n_features = (l_max+1) * len(all_species)**2 * n_max**2

    def forward(self, structures):
        structures = rascaline.torch.systems_to_torch(structures)
        features = self.feature_calculator(structures)
        features = features.keys_to_properties(self.neighbor_labels_1)
        features = features.keys_to_properties(self.neighbor_labels_2)
        # convert features to a dictionary
        # we also need a dictionary to store their structure number
        features_as_dict = {}
        structure_indices = {}
        for species in self.all_species:
            if species not in features.keys["species_center"]:
                features_as_dict[species] = torch.zeros((0, self.n_features))
                structure_indices[species] = torch.zeros((0,), dtype=torch.int)
            else:
                features_as_dict[species] = features.block({"species_center": species}).values
                structure_indices[species] = features.block({"species_center": species}).samples["structure"]

        return features_as_dict, structure_indices

class AtomicNNs(torch.nn.Module):
    def __init__(self, all_species, n_features, n_neurons, n_neurons_last_layer, activation, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_features = n_features
        self.all_species = all_species
        self.atomic_nns = torch.nn.ModuleDict({
            str(species): torch.nn.Sequential(
                torch.nn.LayerNorm(n_features),
                torch.nn.Linear(n_features, n_neurons),
                torch.nn.LayerNorm(n_neurons),
                activation,
                torch.nn.Linear(n_neurons, n_neurons),
                torch.nn.LayerNorm(n_neurons),
                activation,
                torch.nn.Linear(n_neurons, n_neurons_last_layer),
                torch.nn.LayerNorm(n_neurons_last_layer),
                activation
            ) for species in all_species
        })

    def forward(self, features_and_structure_indices):
        features, structure_indices = features_and_structure_indices
        device = next(self.parameters()).device
        return {
            species: self.atomic_nns[str(species)](features[species].to(device)) for species in self.all_species
        }, structure_indices

class SumAtomsModule(torch.nn.Module):

    def __init__(self, all_species, n_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.all_species = all_species
        self.n_species = len(all_species)
        self.n_features = n_features

    def forward(self, features_and_structure_indices):
        features, structure_indices = features_and_structure_indices
        # recover number of structures from structure_indices (as the max of the indices over all species)
        n_structures = 0
        for species in self.all_species:
            if structure_indices[species].shape[0] == 0: continue
            n_structures = max(torch.max(structure_indices[species], dim=0).values.item()+1, n_structures)
        # index add including elements
        summed_features = []
        for species in self.all_species:
            summed_features.append(
                torch.zeros(n_structures, self.n_features, dtype=torch.get_default_dtype(), device=features[species].device).index_add_(
                    source=features[species],
                    index=structure_indices[species].to(features[species].device),
                    dim=0
                )
            )
        summed_features = torch.stack(summed_features, dim=1).reshape(n_structures, self.n_species*self.n_features)
        return summed_features
