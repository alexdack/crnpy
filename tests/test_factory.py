import crnpy
import numpy as np

def test_create_crn():
    species =  np.array(['X_1', 'X_2', 'Y_1'])
    reaction_rates = np.array([5.7, 10.3])
    react_stoch = np.array([[1,0,1], [0,1,1]])
    prod_stoch = np.array([[0,0,2], [0,0,1]])

    crn = crnpy.create_crn(species, reaction_rates, react_stoch, prod_stoch)

    np.testing.assert_array_equal(crn.species, species)
    np.testing.assert_array_equal(crn.reaction_rates, reaction_rates)
    np.testing.assert_array_equal(crn.reaction_stoichiometry, react_stoch)
    np.testing.assert_array_equal(crn.product_stoichiometry, prod_stoch)
    np.testing.assert_array_equal(crn.initial_concentrations, np.array([0,0,0]))
    np.testing.assert_array_equal(crn.number_of_species, 3)
    np.testing.assert_array_equal(crn.number_of_reactions, 2)
    np.testing.assert_array_equal(crn.stoichiometry_matrix, np.array([[-1,0,1], [0,-1,0]]))


