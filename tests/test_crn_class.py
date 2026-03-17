import crnpy
import numpy as np
import pytest
from pathlib import Path
from crnpy.crn.token import create_vocab, parse_tuples_into_matrix
import re

print(crnpy.__file__)

@pytest.fixture
def crn_obj():
    species =  np.array(['X_1', 'X_2', 'Y_1'])
    reaction_rates = np.array([5.7, 10.3])
    react_stoch = np.array([[1,0,1], [0,1,1]])
    prod_stoch = np.array([[0,0,2], [0,0,1]])
    inits = np.array([9.0, 10.6, 11.0])

    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
        "initial_concentrations": inits
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)
    return crn

@pytest.fixture
def crn_obj_same():
    species =  np.array(['X_1', 'X_2', 'Y_1'])
    reaction_rates = np.array([5.7, 10.3])
    react_stoch = np.array([[1,0,1], [0,1,1]])
    prod_stoch = np.array([[0,0,2], [0,0,1]])
    inits = np.array([9.0, 10.6, 11.0])

    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
        "initial_concentrations": inits
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)
    return crn

@pytest.fixture
def crn_text():
    return "#X_1=9.0,X_2=10.6,Y_1=11.0\nX_1 + Y_1->Y_1 + Y_1,5.7\nX_2 + Y_1->Y_1,10.3\n"

@pytest.fixture
def crn_single():
    species =  np.array(['X_1'])
    reaction_rates = np.array([1])
    react_stoch = np.array([[1]])
    prod_stoch = np.array([[0]])
    inits = np.array([10.0])

    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
        "initial_concentrations": inits
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)
    return crn

@pytest.fixture
def crn_double():
    species =  np.array(['X_1'])
    reaction_rates = np.array([0.5, 0.5])
    react_stoch = np.array([[1], [1]])
    prod_stoch = np.array([[0], [0]])
    inits = np.array([10.0])

    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
        "initial_concentrations": inits
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)
    return crn

@pytest.fixture
def crn_verbose():
    species =  np.array(['X_1', 'X_2', 'X_3'])
    reaction_rates = np.array([0.5, 0.5, 0, 0.5])
    react_stoch = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    prod_stoch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0]])
    inits = np.array([10.0, 5.0, 1])

    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
        "initial_concentrations": inits
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)
    return crn

@pytest.fixture
def crn_blowup():
    species =  np.array(['X_1'])
    reaction_rates = np.array([0.5])
    react_stoch = np.array([[1]])
    prod_stoch = np.array([[2]])
    inits = np.array([9.9e5])

    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
        "initial_concentrations": inits
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)
    return crn

def test_crn_repr(crn_obj, crn_text):
    rep = repr(crn_obj)
    assert crn_text == rep

def test_crn_str(crn_obj, crn_text):
    s = str(crn_obj)
    assert crn_text == s

def test_hash(crn_obj, crn_obj_same):
    assert(crn_obj.crn_id == crn_obj_same.crn_id)
    assert hash(crn_obj) == hash(crn_obj_same)
    
def test_integrate(crn_obj):
    _t_length = 0.01
    _t_step = 0.0025

    sol_crn = crn_obj.integrate(_t_length, _t_step)
    
    assert sol_crn.y.shape == (3, 4)
    assert sol_crn.t.shape == (4,)
    
    np.testing.assert_array_equal(sol_crn.t, np.array([0, 0.0025, 0.005, 0.0075]))
    np.testing.assert_allclose(sol_crn.y, np.array([[9.,  7.618279,  6.326681,  5.162753], [10.6,  7.843389,  5.606759,  3.882952], [11., 12.381721, 13.673319, 14.837247]]), rtol=1e-4)

def test_integrate_blowup(crn_blowup):
    _t_length = 2
    _t_step = 0.0025

    sol_crn = crn_blowup.integrate(_t_length, _t_step)
    
    print(sol_crn.y.shape)
    print(sol_crn.t.shape)

    assert sol_crn.y.shape == (1, 9)
    assert sol_crn.t.shape == (9,)
    
def test_distance_from_blowup(crn_obj, crn_blowup):
    _t_length = 2
    _t_step = 0.0025

    (mse, mse_ends, change_in_val) = crn_obj.distance_from(crn_blowup, np.asarray(['X_1']), _t_length, _t_step)
    assert np.isclose(mse, 989961595055.306, atol=1e-4)
    assert np.isclose(mse_ends, 999896244560.6056, atol=1e-4)
    assert np.isclose(change_in_val, 7.455512101724974, atol=1e-4)

    (metric, mse_ends, change_in_val) = crn_obj.distance_from(crn_blowup, np.asarray(['X_1']), _t_length, _t_step, distance_metric='cosine')
    assert np.isclose(change_in_val, 7.455512101724974, atol=1e-4)
    assert np.isclose(metric, 0.11731017300393398, atol=1e-4)  
    assert np.isclose(mse_ends, 999896244560.6056, atol=1e-4)  

  
def test_distance_from_same(crn_obj):
    _t_length = 0.01
    _t_step = 0.0025
    mse, mse_ends, change_in_val = crn_obj.distance_from(crn_obj, np.asarray(['X_1', 'X_2']), _t_length, _t_step)
    assert mse < 1e-9
    metric, mse_ends, change_in_val = crn_obj.distance_from(crn_obj, np.asarray(['X_1', 'X_2']), _t_length, _t_step, distance_metric='cosine')
    assert metric < 1e-9
    mae, mse_ends, change_in_val = crn_obj.distance_from(crn_obj, np.asarray(['X_1', 'X_2']), _t_length, _t_step, distance_metric='mae')
    assert mae < 1e-9



def test_distance_from_net_reactions(crn_single, crn_double):
    _t_length = 0.1
    _t_step = 0.0025
    mse, mse_ends, change_in_val = crn_single.distance_from(crn_double, np.asarray(['X_1']), _t_length, _t_step)
    assert mse < 1e-9
    metric, mse_ends, change_in_val = crn_single.distance_from(crn_double, np.asarray(['X_1']), _t_length, _t_step, distance_metric='cosine')
    assert metric < 1e-9
    mae, mse_ends, change_in_val = crn_single.distance_from(crn_double, np.asarray(['X_1']), _t_length, _t_step, distance_metric='mae')
    assert mae < 1e-9

def test_distance_from_far(crn_single, crn_obj):
    _t_length = 0.1
    _t_step = 0.0025
    mse, mse_ends, change_in_val = crn_single.distance_from(crn_obj, np.asarray(['X_1']), _t_length, _t_step)
    assert mse > 1e1

def test_tokenize(crn_obj):
    max_number_of_species = 4
    max_number_of_reaction = 3
    vocab, inv_vocab = create_vocab(max_number_of_species, 2)
    stoichiometry_tokens, rate_tokens, initial_concentrations_tokens = crn_obj.tokenize(vocab, max_number_of_species, max_number_of_reaction) 
    print(stoichiometry_tokens)
    print(rate_tokens)
    print(initial_concentrations_tokens)
    np.testing.assert_array_equal(crn_obj.reaction_rates, rate_tokens[:-1])
    assert rate_tokens[-1] == 0 
    assert stoichiometry_tokens[max_number_of_reaction-1] == 0 
    assert initial_concentrations_tokens[-1] == 0 

    np.testing.assert_array_equal(crn_obj.initial_concentrations, initial_concentrations_tokens[:-1])
    np.testing.assert_array_equal(np.array([117, 153, 0]), stoichiometry_tokens)
    
    # checks that you can convert backwards to the same stoichiometry
    react_stoich, product_stoich = parse_tuples_into_matrix(stoichiometry_tokens, inv_vocab, crn_obj.number_of_reactions, crn_obj.number_of_species)
    np.testing.assert_array_equal(crn_obj.reaction_stoichiometry, react_stoich)
    np.testing.assert_array_equal(crn_obj.product_stoichiometry, product_stoich)

def test_reduce(crn_verbose):
    print(crn_verbose)
    assert crn_verbose.number_of_reactions == 4
    assert crn_verbose.stoichiometry_matrix.shape == (4, 3)
    crn_verbose.reduce()
    print(crn_verbose)
    assert crn_verbose.number_of_reactions == 1
    assert crn_verbose.stoichiometry_matrix.shape == (1, 1)

def test_steady_states(crn_obj):
    vars, ode_system = crn_obj.steady_states_eqs()
    print(vars)
    print(ode_system)
    expected_vars = "X_1 X_2 Y_1"
    expected_ode_system = '-5.7*X_1*Y_1, -10.3*X_2*Y_1, +5.7*X_1*Y_1, 0.7071067811865475*X_1 + 0.7071067811865476*Y_1-14.142135623730951'
    assert vars == expected_vars

    # Extract floats
    actual_floats = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', ode_system)]
    expected_floats = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', expected_ode_system)]

    assert 6 == len(expected_floats)

    # Compare each float with a relative or absolute tolerance
    for actual_float, expected_float in zip(actual_floats, expected_floats):    
        # Compare with a relative or absolute tolerance
        assert np.isclose(actual_float, expected_float, atol=1e-6)

