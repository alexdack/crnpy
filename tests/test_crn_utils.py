from crnpy.crn import utils
import numpy as np
from pathlib import Path
import pytest

@pytest.fixture
def csv_path():
    TEST_DIR = Path(__file__).parent
    DATA_DIR = TEST_DIR / "data"
    csv_path = DATA_DIR / "test_1.csv"
    return csv_path

@pytest.fixture
def crn_to_save():
    TEST_DIR = Path(__file__).parent
    DATA_DIR = TEST_DIR / "data"
    csv_path = DATA_DIR / "crn_to_save.txt"
    return csv_path

@pytest.fixture
def test_crn():
    TEST_DIR = Path(__file__).parent
    DATA_DIR = TEST_DIR / "data"
    csv_path = DATA_DIR / "test_crn.txt"
    return csv_path

def test_open_csv(csv_path):
    np.testing.assert_array_equal(utils.open_csv(csv_path), np.array([[1., 1.]]))

def test_save_crn(crn_to_save):
    assert True

def test_read_crn_txt(test_crn):

    species, reaction_rates, react_stoch, prod_stoch, stoch_mat, number_species, number_reactions, initial_concs_vec = utils.read_crn_txt(test_crn)

    _species = np.asarray(species)
    assert np.isin('X_1', _species)
    assert np.isin('Y_1', _species)
    assert np.isin('X_2', _species)
    assert not np.isin('Z_1', _species)
    np.testing.assert_array_equal(_species, np.array(['X_1', 'X_2', 'Y_1']))
    np.testing.assert_array_equal(reaction_rates, np.array([5.7, 10.3]))
    np.testing.assert_array_equal(react_stoch, np.array([[1,0,1], [0,1,1]]))
    np.testing.assert_array_equal(prod_stoch, np.array([[0,0,2], [0,0,1]]))
    np.testing.assert_array_equal(stoch_mat, np.array([[-1,0,1], [0,-1,0]]))
    assert number_species == 3
    assert number_reactions == 2
    np.testing.assert_array_equal(initial_concs_vec, np.array([9.0, 10.6, 11.0]))

def test_stoch_mat_to_mass_action():
    _t = 0
    _x = np.array([9.0, 10.6, 11.0])
    _reaction_rates = np.array([5.7, 10.3])
    _react_stoch = np.array([[1,0,1], [0,1,1]])
    _stoch_mat = np.array([[-1,0,1], [0,-1,0]])
    mass_action = utils.stoch_mat_to_mass_action(_t, _x, _reaction_rates, _react_stoch, _stoch_mat)

    _dx1 = -9.0*11.0*5.7
    _dx2 = -10.3*10.6*11.0
    _dy1 = 9.0*11.0*5.7

    np.testing.assert_array_equal(mass_action, np.array([_dx1, _dx2, _dy1]))


def test_simulate_trajectory(test_crn):
    _t_length = 0.01
    _t_step = 0.0025
    sol_crn = utils.simulate_trajectory(test_crn, _t_length, _t_step)
    assert sol_crn.y.shape == (3, 4)
    assert sol_crn.t.shape == (4,)
    np.testing.assert_array_equal(sol_crn.t, np.array([0, 0.0025, 0.005, 0.0075]))
    np.testing.assert_allclose(sol_crn.y, np.array([[9.,  7.618279,  6.326681,  5.162753], [10.6,  7.843389,  5.606759,  3.882952], [11., 12.381721, 13.673319, 14.837247]]))

def test_convert_arrays_to_crn_text():
    _species = np.array(['X_1', 'X_2', 'Y_1'])
    _reaction_rates = np.array([5.7, 10.3])
    _react_stoch = np.array([[1,0,1], [0,1,1]])
    _prod_stoch = np.array([[0,0,2], [0,0,1]])
    _inits = np.array([9.0, 10.6, 11.0])

    s = utils.convert_arrays_to_crn_text(_species, _reaction_rates, _react_stoch, _prod_stoch, _inits)
    assert s == "#X_1=9.0,X_2=10.6,Y_1=11.0\nX_1 + Y_1->Y_1 + Y_1,5.7\nX_2 + Y_1->Y_1,10.3\n"


