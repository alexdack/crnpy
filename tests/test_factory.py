import crnpy
import numpy as np
import pytest
from pathlib import Path
from crnpy.crn import token

@pytest.fixture
def test_crn():
    TEST_DIR = Path(__file__).parent
    DATA_DIR = TEST_DIR / "data"
    csv_path = DATA_DIR / "test_crn.txt"
    return csv_path


def test_create_crn_from_arrays():
    species =  np.array(['X_1', 'X_2', 'Y_1'])
    reaction_rates = np.array([5.7, 10.3])
    react_stoch = np.array([[1,0,1], [0,1,1]])
    prod_stoch = np.array([[0,0,2], [0,0,1]])

    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)

    np.testing.assert_array_equal(crn.species, species)
    np.testing.assert_array_equal(crn.reaction_rates, reaction_rates)
    np.testing.assert_array_equal(crn.reaction_stoichiometry, react_stoch)
    np.testing.assert_array_equal(crn.product_stoichiometry, prod_stoch)
    np.testing.assert_array_equal(crn.initial_concentrations, np.array([0,0,0]))
    np.testing.assert_array_equal(crn.number_of_species, 3)
    np.testing.assert_array_equal(crn.number_of_reactions, 2)
    np.testing.assert_array_equal(crn.stoichiometry_matrix, np.array([[-1,0,1], [0,-1,0]]))

    print(crn)

def test_create_crn_from_file(test_crn):

    _from_file = {
        "filename": test_crn
    }

    crn = crnpy.create_crn(from_file=_from_file)

    np.testing.assert_array_equal(crn.species, np.array(['X_1', 'X_2', 'Y_1']))
    np.testing.assert_array_equal(crn.reaction_rates, np.array([5.7, 10.3]))
    np.testing.assert_array_equal(crn.reaction_stoichiometry, np.array([[1,0,1], [0,1,1]]))
    np.testing.assert_array_equal(crn.product_stoichiometry, np.array([[0,0,2], [0,0,1]]))
    np.testing.assert_array_equal(crn.initial_concentrations, np.array([9.0, 10.6, 11.0]))
    np.testing.assert_array_equal(crn.number_of_species, 3)
    np.testing.assert_array_equal(crn.number_of_reactions, 2)
    np.testing.assert_array_equal(crn.stoichiometry_matrix, np.array([[-1,0,1], [0,-1,0]])) 
    print(crn)

def test_create_crn_from_random():
    np.random.seed(0)
    _from_random = {
        "number_of_species": 3,
        "number_of_reactions": 2
    }

    crn = crnpy.create_crn(from_random=_from_random)
    print(crn.product_stoichiometry)
    print(crn.initial_concentrations)

    np.testing.assert_array_equal(crn.species, np.array(['S_1', 'S_2', 'S_3']))
    np.testing.assert_allclose(crn.reaction_rates, np.array([5.836039, 1.492059]), rtol=1e-3)
    np.testing.assert_array_equal(crn.reaction_stoichiometry, np.array([[1, 0, 0], [0, 1, 1]]))
    np.testing.assert_array_equal(crn.product_stoichiometry, np.array([[0, 0, 1], [2, 0, 0]]))
    np.testing.assert_allclose(crn.initial_concentrations, np.array([2.66109578, 9.40172515, 6.47247125]), rtol=1e-3)
    np.testing.assert_array_equal(crn.number_of_species, 3)
    np.testing.assert_array_equal(crn.number_of_reactions, 2)
    np.testing.assert_array_equal(crn.stoichiometry_matrix, np.array([[-1,0,1], [2,-1,-1]])) 

    _from_random_bi_only = {
        "number_of_species": 3,
        "number_of_reactions": 2,
        "reaction_molecularity_ratio": {0: 0, 1: 0, 2: 1},
        "product_molecularity_ratio": {0: 0, 1: 0, 2: 1}
    }

    crn = crnpy.create_crn(from_random=_from_random_bi_only)
    np.testing.assert_array_equal(crn.reaction_stoichiometry, np.array([[2, 0, 0], [1, 0, 1]]))
    np.testing.assert_array_equal(crn.product_stoichiometry, np.array([[0, 0, 2], [0, 2, 0]]))

    
    print(crn)

def test_create_crn_from_tokens():
    number_of_species = 3
    vocab, inv_vocab = token.create_vocab(number_of_species, 2)
    stoichiometry_tokens = [69, 83]
    rate_tokens = np.asarray([1, 10])
    initial_concentrations_tokens = np.asarray([10, 0, 0])

    crn = crnpy.create_crn(from_tokens={
        "stoichiometry_tokens": stoichiometry_tokens,
        "rate_tokens": rate_tokens,
        "initial_concentrations_tokens": initial_concentrations_tokens, 
        "inv_vocab": inv_vocab
    })
    np.testing.assert_array_equal(crn.reaction_stoichiometry, np.array([[1,0,1], [0,1,1]]))
    np.testing.assert_array_equal(crn.product_stoichiometry, np.array([[0,0,2], [0,0,1]]))
    print(crn)

