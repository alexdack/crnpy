import crnpy
import numpy as np
import pytest
from pathlib import Path

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

