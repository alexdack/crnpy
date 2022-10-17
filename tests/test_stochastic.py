import numpy as np;
from crnpy import stochastic
import random;

def test_crn_state_rates_generator():
    random.seed(10)
    start_state = np.array([20]);
    reactant_matrix = np.array([[0,1]]);
    product_matrix = np.array([[1,0],]);
    reaction_rates = np.array([1]);
    null_index = 1;

    output = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index)
    result =  (np.array([[21]]), np.array([1.]));
    assert output == result

def test_gillespie_simulation():
    random.seed(10)
    start_state = np.array([20]);
    reactant_matrix = np.array([[0,1]]);
    product_matrix = np.array([[1,0],]);
    reaction_rates = np.array([1]);
    null_index = 1;
    tRun = 3;

    output = stochastic.gillespie_simulation( tRun, start_state, reactant_matrix, product_matrix, reaction_rates, null_index )
    result =  (np.array([0.        , 0.84723725, 1.71020359, 3.38856965]),
               np.array([[20],
                         [21],
                         [22],
                         [23]]))

    time = abs(output[0] - result[0]) < 0.0000001;
    states = output[1] == result[1];
    assert time.all()
    assert states.all()