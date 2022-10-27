import numpy as np;
from crnpy import stochastic
import scipy.special as spc

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

def test_compute_stationary_distribution():
    np.random.seed(5);
    steady_state = np.array([10]);
    reactant_matrix = np.array([[1, 0],[0, 1]]);
    product_matrix = np.array([[0, 1],[1, 0],]);
    reaction_rates = np.array([0.1, 1]);
    n_max = 23;
    null_index = 1;
    number_of_trajectories = 200;
    n,p = stochastic.compute_stationary_distribution(steady_state, n_max, number_of_trajectories, reactant_matrix, product_matrix, reaction_rates, null_index);

    pCME = 1/spc.factorial(n)*pow(10,n)*np.exp(-10);

    test_range = n == np.arange(0,n_max,1);
    assert test_range.all();
    test_p = abs(np.transpose(p[:,0])- pCME) < 0.05;
    assert test_p.all();

def compute_stationary_distribution_single_traj(steady_state, n_max, tFinal, timeStep, reactant_matrix, product_matrix, reaction_rates, null_index):
    np.random.seed(5);
    steady_state = np.array([10]);
    reactant_matrix = np.array([[1, 0],[0, 1]]);
    product_matrix = np.array([[0, 1],[1, 0],]);
    reaction_rates = np.array([0.1, 1]);
    n_max = 23;
    null_index = 1;
    tFinal = 1000;
    timeStep = 1;
    n,p = stochastic.compute_stationary_distribution_single_traj(steady_state, n_max, tFinal, timeStep, reactant_matrix, product_matrix, reaction_rates, null_index);

    test_range = n == np.arange(0,n_max,1);
    assert test_range.all();

    pCME = 1/spc.factorial(n)*pow(10,n)*np.exp(-10);

    test_p = abs(np.transpose(p[:,0])- pCME) < 0.001;
    assert test_p.all();
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
    result = stochastic.interpolate_all_trajectories(test_trajs, 0.5);
    expected = (np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5]),
                [(np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5]),
                  np.array([[0., 1.],
                         [0., 1.],
                         [0., 1.],
                         [1., 1.],
                         [1., 1.],
                         [0., 1.],
                         [0., 1.],
                         [1., 1.]])),
                 (np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. ]),
                  np.array([[0., 2.],
                         [0., 2.],
                         [0., 2.],
                         [2., 2.],
                         [2., 2.],
                         [1., 2.],
                         [1., 2.],
                         [1., 2.],
                         [1., 2.]]))])
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

def test_resample_to_fixed_step():
    time_in = np.array([   0.        ,    2.6862601 ,    5.30086411,  5.5,
                           7, 10]);

    traj = np.array([[0,1],
                     [1,2],
                     [2,4],
                     [8, 3],
                     [7, 8],
                     [8, 9]]);

    timeStep = 1;
    time_arr, state_arr = stochastic.resample_to_fixed_step(time_in, traj, timeStep );

    exp_state = np.array([[0., 1.],
                       [0., 1.],
                       [0., 1.],
                       [1., 2.],
                       [1., 2.],
                       [1., 2.],
                       [8., 3.],
                       [8., 3.],
                       [7., 8.],
                       [7., 8.]]);

    exp_time = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    test_state = state_arr == exp_state;
    test_time = time_arr == exp_time;

    assert test_state.all();
    assert test_time.all();

def test_crn_state_rates_generator_1():
    # 1) kv: null -> A, alpha = kv
    np.random.seed(5)
    kv = 1;
    A0 = 30;
    reactant_matrix = np.array([[0, 1]]);
    product_matrix = np.array([[1, 0]]);
    null_index = 1;
    start_state = np.array([A0]);
    reaction_rates = np.array([kv]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == kv)
    assert(state[0][0] == A0+1)

def test_crn_state_rates_generator_2():
    # 2) k: A -> null, alpha = A*k
    np.random.seed(5)
    k = 1;
    A0 = 30;
    reactant_matrix = np.array([[1, 0]]);
    product_matrix = np.array([[0, 1]]);
    null_index = 1;
    start_state = np.array([A0]);
    reaction_rates = np.array([k]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == k*A0)
    assert(state[0][0] == A0-1)

def test_crn_state_rates_generator_3():
    # 3) kbyv: A + B -> null, alpha = A*B*kbyv
    np.random.seed(5)
    kbyv = 1;
    A0 = 30;
    B0 = 10;
    reactant_matrix = np.array([[1, 1, 0]]);
    product_matrix = np.array([[0, 0, 1]]);
    null_index = 2;
    start_state = np.array([A0, B0 ]);
    reaction_rates = np.array([kbyv]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == kbyv*A0*B0);
    assert(state[0][0] == A0-1);
    assert(state[0][1] == B0-1);

def test_crn_state_rates_generator_4():
    # 4) kbyv: A + A -> null, alpha = A*(A-1)* kbyv
    np.random.seed(5)
    kbyv = 1;
    A0 = 30;
    reactant_matrix = np.array([[2, 0]]);
    product_matrix = np.array([[0, 1]]);
    null_index = 1;
    start_state = np.array([A0 ]);
    reaction_rates = np.array([kbyv]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == kbyv*A0*(A0-1));
    assert(state[0][0] == A0-2);

def test_crn_state_rates_generator_5():
    # 5) kbyv2: A + B + C -> null A*B*C*kbyv2
    np.random.seed(5)
    kbyv2 = 1;
    A0 = 30;
    B0 = 20;
    C0 = 10;
    reactant_matrix = np.array([[1, 1, 1, 0]]);
    product_matrix = np.array([[0, 0, 0, 1]]);
    null_index = 3;
    start_state = np.array([A0, B0, C0]);
    reaction_rates = np.array([kbyv2]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == kbyv2*A0*B0*C0)
    assert(state[0][0] == A0-1)
    assert(state[0][1] == B0-1)
    assert(state[0][2] == C0-1)

def test_crn_state_rates_generator_6():
    # 6) kbyv2: 2A + B -> null, A*(A-1)*B*kbyv2
    np.random.seed(5)
    kbyv2 = 1;
    A0 = 30;
    B0 = 20;
    reactant_matrix = np.array([[2, 1, 0]]);
    product_matrix = np.array([[0, 0, 1]]);
    null_index = 2;
    start_state = np.array([A0, B0]);
    reaction_rates = np.array([kbyv2]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == kbyv2*A0*(A0-1)*B0)
    assert(state[0][0] == A0-2)
    assert(state[0][1] == B0-1)

def test_crn_state_rates_generator_7():
    # 7) kbyv2: 3A -> null, A*(A-1)*(A-2)*kbyv2
    np.random.seed(5)
    kbyv2 = 1;
    A0 = 30;
    reactant_matrix = np.array([[3, 0]]);
    product_matrix = np.array([[0, 1]]);
    null_index = 1;
    start_state = np.array([A0]);
    reaction_rates = np.array([kbyv2]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == kbyv2*A0*(A0-1)*(A0-2))
    assert(state[0][0] == A0-3)

def test_crn_state_rates_generator_8():
    # 8) kbyv2: 3A -> A, A*(A-1)*(A-2)*kbyv2
    np.random.seed(5)
    kbyv2 = 1;
    A0 = 30;
    reactant_matrix = np.array([[3, 0]]);
    product_matrix = np.array([[1, 0]]);
    null_index = 1;
    start_state = np.array([A0]);
    reaction_rates = np.array([kbyv2]);

    state, propensity = stochastic.crn_state_rates_generator(start_state, reactant_matrix, product_matrix, reaction_rates, null_index);

    assert(propensity[0] == kbyv2*A0*(A0-1)*(A0-2))
    assert(state[0][0] == A0-2)