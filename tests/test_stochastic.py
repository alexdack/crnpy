import numpy as np;
from crnpy import stochastic
import random;

def test_crn_state_rates_generator():
    np.random.seed(10)
    start_state = np.array([20]);
    reactant_matrix = np.array([[0,1]]);
    product_matrix = np.array([[1,0],]);
    reaction_rates = np.array([1]);
    null_index = 1;

    output = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index)
    result =  (np.array([[21]]), np.array([1.]));
    assert output == result

def test_gillespie_simulation_prod_decay_CRN():
    np.random.seed(5);
    start_state = np.array([0]);
    reactant_matrix = np.array([[1, 0],[0, 1]]);
    product_matrix = np.array([[0, 1],[1, 0],]);
    reaction_rates = np.array([0.1, 1]);
    tRun = 3;
    null_index = 1;

    output = stochastic.gillespie_simulation( tRun, start_state, reactant_matrix, product_matrix, reaction_rates, null_index )
    result =  (np.array([0.        , 1.50510866, 2.93819424, 3.53535893]),
               np.array([[0],
                         [1],
                         [2],
                         [3]]))

    time = abs(output[0] - result[0]) < 0.0000001;
    states = output[1] == result[1];
    assert time.all()
    assert states.all()

def test_gillespie_simulation_prod_decay_final_only():
    np.random.seed(5);
    start_state = np.array([0]);
    reactant_matrix = np.array([[1, 0],[0, 1]]);
    product_matrix = np.array([[0, 1],[1, 0],]);
    reaction_rates = np.array([0.1, 1]);
    tRun = 3;
    null_index = 1;

    output = stochastic.gillespie_simulation( tRun, start_state, reactant_matrix, product_matrix, reaction_rates, null_index, final_only=True)
    result =  (3.53535893,
               np.array([[3]]));

    time = abs(output[0] - result[0]) < 0.0000001;
    states = output[1] == result[1];
    assert time.all()
    assert states.all()

def test_find_smallest_time_between_events():
    t1 = np.array([0, 1, 2 ,3, 4]);
    t3 = np.array([0, 1, 2, 4]);
    x = np.array([0, 1, 0, 1, 2]);
    t2 = np.array([0, 0.5, 1, 1.5, 2, 2.5 ,3, 3.5, 4]);
    test_trajs = [(t1, np.array([[0], [1], [0], [1], [2] ]) ),
              (t3, np.array([[0], [2], [1], [2] ]) )];
    res = stochastic.find_smallest_time_between_events(test_trajs);
    expected = 1;
    assert expected == res;

def test_find_smallest_final_event_time():
    t1 = np.array([0, 1, 2 ,3, 4]);
    t3 = np.array([0, 1, 2, 4.1]);
    test_trajs = [(t1, np.array([[0], [1], [0], [1], [2] ]) ),
                  (t3, np.array([[0], [2], [1], [2] ]) )];
    res = stochastic.find_smallest_final_event_time(test_trajs);
    expected = 4;
    assert expected == res;

def test_interpolate_onto_same_dt():
    old_time_axis = np.array([0, 1, 2 ,3, 4]);
    trajectory = np.array([0, 1, 0, 1, 2]);
    resampled_time_axis = np.array([0, 0.5, 1, 1.5, 2, 2.5 ,3, 3.5, 4]);
    result = stochastic.interpolate_onto_same_dt( resampled_time_axis, old_time_axis, trajectory );
    expected = np.array([0., 0., 1., 1., 0., 0., 1., 1., 2]);
    res = expected == result;
    assert res.all();

def test_interpolate_all_states():
    t1 = np.array([0, 1, 2 ,3, 4]);
    t3 = np.array([0, 1, 2, 4.1]);
    test_trajs = [(t1, np.array([[0], [1], [0], [1], [2] ]) ),
              (t3, np.array([[0], [2], [1], [2] ]) )];
    resampled_time_axis = np.array([0, 0.5, 1, 1.5, 2, 2.5 ,3, 3.5, 4]);
    result = stochastic.interpolate_all_states(test_trajs[0], resampled_time_axis);
    expected = [np.array([0., 0., 1., 1., 0., 0., 1., 1., 2.])];
    res = expected[0] == result[0];
    assert res.all();

def test_interpolate_all_trajectories():
    t1 = np.array([0, 1, 2 ,3, 4]);
    t3 = np.array([0, 1, 2, 4.1]);
    test_trajs = [(t1, np.array([[0,1], [1,1], [0,1], [1,1], [2,1] ]) ),
              (t3, np.array([[0,2], [2,2], [1,2], [2,2] ]) )];
    result = stochastic.interpolate_all_trajectories(test_trajs);
    expected = (np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5]),
                [[np.array([0., 0., 1., 1., 0., 0., 1., 1.]),
                  np.array([1., 1., 1., 1., 1., 1., 1., 1.])],
                 [np.array([0., 0., 2., 2., 1., 1., 1., 1.]),
                  np.array([2., 2., 2., 2., 2., 2., 2., 2.])]])
    time = expected[0] == result[0];
    assert time.all();
    states_1 = expected[1][0][0] == result[1][0][0];
    assert states_1.all();
    states_2 = expected[1][0][1] == result[1][0][1];
    assert states_2.all();
    states_3 = expected[1][1][0] == result[1][1][0];
    assert states_3.all();
    states_4 = expected[1][1][1] == result[1][1][1];
    assert states_4.all();

