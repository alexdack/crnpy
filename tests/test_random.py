from crnpy.crn import random
import numpy as np

def test_iterate_all_molecularity_tuples():
    res_1 = random.iterate_all_molecularity_tuples(2, 1)
    assert res_1[0] == [()]
    assert res_1[1] == [(0,), (1,)]

    res_2 = random.iterate_all_molecularity_tuples(3, 3)
    assert res_2[0] == [()]
    assert res_2[1] == [(0,), (1,), (2,)]
    assert res_2[2] == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    print(res_2[3])
    assert res_2[3] == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (0, 2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]

def test_create_stoichiometry_matrices():
    np.random.seed(0)
    number_of_reactions = 10000
    number_of_species = 3
    molecularity_ratio = {0:1, 1:1}
    res_unimolecular = random.create_stoichiometry_matrices(number_of_reactions, number_of_species, molecularity_ratio)
    perc_split = np.sum(np.sum(res_unimolecular))/number_of_reactions
    assert perc_split == 0.4936

    molecularity_ratio_bi = {0: 1, 1:1, 2:1}
    number_of_reactions_bi = 5
    number_of_species_bi = 3

    res_bimolecular_only = random.create_stoichiometry_matrices(number_of_reactions_bi, number_of_species_bi, molecularity_ratio_bi)
    print(res_bimolecular_only)
    np.testing.assert_array_equal(res_bimolecular_only, np.array([[0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 2], [0, 0, 0]]))
