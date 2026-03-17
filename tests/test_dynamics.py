import crnpy
import numpy as np
import pytest
from pathlib import Path
from crnpy.crn import dynamics
import re

@pytest.fixture
def test_crn():
    species =  np.array(['X_1', 'X_2', 'Y_1'])
    reaction_rates = np.array([5.7, 10.3])
    react_stoch = np.array([[1,0,1], [0,1,1]])
    prod_stoch = np.array([[0,0,2], [0,0,1]])
    inits = np.array([1, 1, 1])
    _from_arrays = {
        "species": species,
        "reaction_rates": reaction_rates,
        "reaction_stoichiometry": react_stoch,
        "product_stoichiometry":prod_stoch,
        "initial_concentrations": inits
    }

    crn = crnpy.create_crn(from_arrays=_from_arrays)
    return crn


def test_compute_conservation_laws(test_crn):
    null_space_matrix, C = dynamics.compute_conservation_laws(test_crn.stoichiometry_matrix, test_crn.initial_concentrations)

    # Check that the null space matrix has the correct shape
    assert null_space_matrix.shape[0] == test_crn.number_of_species
    assert null_space_matrix.shape[1] == test_crn.number_of_species - np.linalg.matrix_rank(test_crn.stoichiometry_matrix)

    # Check that C has the correct shape
    assert C.shape[0] == null_space_matrix.shape[1]

    # Check that the null space matrix is a null space of the stoichiometry matrix
    np.testing.assert_allclose(test_crn.stoichiometry_matrix @ null_space_matrix, np.array([[0],[0]]), atol=1e-8)

def test_vars_julia_solver(test_crn):
    vars_str = dynamics.vars_julia_solver(test_crn.species)
    expected_str = " ".join(test_crn.species)
    assert vars_str == expected_str

def test_f_mass_action_julia_solver(test_crn):
    mass_action = dynamics.f_mass_action_julia_solver(test_crn.species, test_crn.reaction_rates, test_crn.reaction_stoichiometry, test_crn.stoichiometry_matrix)
    expected_mass_action = ['-5.7*X_1*Y_1', '-10.3*X_2*Y_1', '+5.7*X_1*Y_1']
    assert mass_action == expected_mass_action

def test_f_conservation_julia_solver(test_crn):
    conservation_laws = dynamics.f_conservation_julia_solver(test_crn.species, test_crn.stoichiometry_matrix, test_crn.initial_concentrations)
    print(conservation_laws)
    expected_conservation_laws = ['0.7071067811865475*X_1 + 0.7071067811865476*Y_1-1.414213562373095']

    # Extract floats
    actual_floats = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', conservation_laws[0])]
    expected_floats = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', expected_conservation_laws[0])]

    # Compare each float with a relative or absolute tolerance
    for actual_float, expected_float in zip(actual_floats, expected_floats):    
        # Compare with a relative or absolute tolerance
        assert np.isclose(actual_float, expected_float, atol=1e-6)

    assert conservation_laws == expected_conservation_laws

def test_steady_states_julia_solver(test_crn):
    vars, ode_system = dynamics.steady_states_julia_solver(test_crn.species, test_crn.reaction_rates, test_crn.reaction_stoichiometry, test_crn.stoichiometry_matrix, test_crn.initial_concentrations)
    assert vars == 'X_1 X_2 Y_1'

    expected_ode_system = '-5.7*X_1*Y_1, -10.3*X_2*Y_1, +5.7*X_1*Y_1, 0.7071067811865475*X_1 + 0.7071067811865476*Y_1-1.414213562373095'

    # Extract floats
    actual_floats = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', ode_system)]
    expected_floats = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', expected_ode_system)]

    assert 6 == len(expected_floats)

    # Compare each float with a relative or absolute tolerance
    for actual_float, expected_float in zip(actual_floats, expected_floats):    
        # Compare with a relative or absolute tolerance
        assert np.isclose(actual_float, expected_float, atol=1e-6)
