import numpy as np
import itertools

def iterate_all_molecularity_tuples(number_of_species:int, molecularity:int):
    positions = list(range(number_of_species))
    all_combinations = {}

    for n in range(molecularity+1):
        all_combinations[n] = list(itertools.combinations_with_replacement(positions, n))
    
    return all_combinations


def create_stoichiometry_matrices(number_of_reactions:int , number_of_species:int , molecularity_ratio: dict):
    # inits an empty matrix
    mat = np.zeros((number_of_reactions, number_of_species))

    # extracts molecularity data
    molecularities = np.asarray(list(molecularity_ratio.keys()))
    molecularity_pairs = iterate_all_molecularity_tuples(number_of_species, np.max(molecularities))

    ratios = np.asarray(list(molecularity_ratio.values()))

    # normalises the ratios
    ratio_sum = np.sum(ratios)
    norm_ratios = np.divide(ratios, ratio_sum)
    indices = np.arange(len(ratios))

    # randomly selects the reaction type of each reaction according to the ratios 
    selected_molecularities_idx = np.random.choice(indices, size=number_of_reactions, p=norm_ratios)
    selected_molecularities = np.asarray([molecularities[idx] for idx in selected_molecularities_idx])

    # populates the matrix
    for r in range(number_of_reactions):
        selected_molecularity_pairs = molecularity_pairs[selected_molecularities[r]]
        selected_molecularity_pairs_idx = len(selected_molecularity_pairs)
        selected_species_idx = np.random.choice(selected_molecularity_pairs_idx)
        for s in selected_molecularity_pairs[selected_species_idx]:
            mat[r,s] += 1

    return mat.astype(np.int32)
