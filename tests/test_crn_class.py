import crnpy
import numpy as np
import pytest
from pathlib import Path

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

def test_crn_repr(crn_obj, crn_text):
    rep = repr(crn_obj)
    assert crn_text == rep

def test_crn_str(crn_obj, crn_text):
    s = str(crn_obj)
    assert crn_text == s

def test_hash(crn_obj, crn_obj_same):
    assert(crn_obj.crn_id == crn_obj_same.crn_id)
    assert hash(crn_obj) == hash(crn_obj_same)
    
