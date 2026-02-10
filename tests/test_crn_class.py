import crnpy
import numpy as np
import pytest
from pathlib import Path
from crnpy.crn.token import create_vocab, parse_tuples_into_matrix
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
    assert mse == np.inf
    assert mse_ends == np.inf
    assert change_in_val == np.inf

  
def test_distance_from_same(crn_obj):
    _t_length = 0.01
    _t_step = 0.0025
    mse, mse_ends, change_in_val = crn_obj.distance_from(crn_obj, np.asarray(['X_1', 'X_2']), _t_length, _t_step)
    assert mse < 1e-9

def test_distance_from_net_reactions(crn_single, crn_double):
    _t_length = 0.1
    _t_step = 0.0025
    mse, mse_ends, change_in_val = crn_single.distance_from(crn_double, np.asarray(['X_1']), _t_length, _t_step)
    assert mse < 1e-9

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
