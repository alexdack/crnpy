import numpy as np;
import math;
from crnpy import tools;
from scipy.interpolate import interp1d;

def crn_state_rates_generator(current_state, reactant_matrix, product_matrix, reaction_rates, null_index):
    # current_state is np.array of molecule numbers for species
    # reactant_matrix is a np.array with the stoichometry values of the reactants. columns = number of species, rows = number of reations.
    # product_matrix is a np.array with the stoichometry values of the reactants. columns = number of species, rows = number of reations.
    # reaction_rates is np.array with one column that correspond to the rate of each reaction

    current_state_w_null = np.append(current_state, 0);
    stoichometry_matrix = product_matrix - reactant_matrix;

    scaled_rates = np.zeros(reaction_rates.shape)

    for i in range(reaction_rates.size):
        scaled_rates[i] = reaction_rates[i] * tools.compute_stoichiometry_terms_stochastic_propensity( reactant_matrix[i,:], current_state_w_null, null_index)

    nonnegative_rates_mask = scaled_rates > 0;


    future_states = np.delete( stoichometry_matrix, null_index, 1) + current_state;
    nonnegative_states_mask = np.all(future_states >= 0, 1);


    possible_states_mask = np.logical_and(nonnegative_states_mask, nonnegative_rates_mask);

    possible_states = future_states[possible_states_mask];
    propensity_arr = scaled_rates[possible_states_mask];

    return (possible_states, propensity_arr )


def gillespie_simulation( tRun, start_state, reactant_matrix, product_matrix, reaction_rates, null_index, final_only=False ):
    t = 0;
    current_state = start_state;

    time_arr = np.array([0]);
    state_arr = np.array([start_state]);


    while t < tRun:
        u, v = crn_state_rates_generator(current_state, reactant_matrix, product_matrix, reaction_rates, null_index);
        k = np.sum(v);
        r = np.random.rand(2,1);
        unbiased_number_time_step = r[0];
        dt = - np.log(unbiased_number_time_step ) / k;
        unbiased_number_state_change = r[1];
        rate_comparison = unbiased_number_state_change * k;
        i_new = 0;
        s = 0;
        for l in range(v.size):
            s = s + v[l];
            if s > rate_comparison:
                i_new = u[l];
                break;
        current_state = i_new;
        t = t + dt;
        if not final_only:
            time_arr = np.append(time_arr, t);
            state_arr = np.append(state_arr, np.array([i_new]),axis=0);
    if final_only:
        time_arr = t;
        state_arr = np.array([current_state]);

    return (time_arr, state_arr)

def find_smallest_time_between_events(list_of_trajectories):

    def deltas(x_tuple):
        return np.min( np.diff(x_tuple[0]) )
    result = np.min( list( map(deltas, list_of_trajectories) ) );

    return result

def compute_stationary_distribution(steady_state, n_max, number_of_trajectories, reactant_matrix, product_matrix, reaction_rates, null_index):
    p = np.zeros([n_max, steady_state.shape[0] ]);
    for i in range(0, number_of_trajectories):
        time, trajectory = gillespie_simulation( i, steady_state, reactant_matrix, product_matrix, reaction_rates, null_index , final_only=True);
        for state_index in range(steady_state.shape[0]):
            final_value = trajectory[state_index]
            if final_value >= 0 and final_value < n_max:
                p[int(final_value), state_index] = p[int(final_value), state_index] + 1;
    p = p/number_of_trajectories;
    n = np.arange(0,n_max,1);
    return (n,p)

def compute_stationary_distribution_single_traj(steady_state, n_max, tFinal, timeStep, reactant_matrix, product_matrix, reaction_rates, null_index):
    p = np.zeros([n_max, steady_state.shape[0] ]);
    time, trajectory = gillespie_simulation( tFinal, steady_state, reactant_matrix, product_matrix, reaction_rates, null_index );
    time_arr, state_arr = resample_to_fixed_step(time, trajectory, timeStep );
    number_of_savings  = time_arr.shape[0];
    for state_index in range(steady_state.shape[0]):
        vec = state_arr[:,state_index].astype(int)
        count_arr = np.zeros(n_max);
        bin_count = np.bincount(vec);
        count_arr[0:bin_count.shape[0]] = np.bincount(vec);
        p_state = count_arr / number_of_savings;
        p[:,state_index] = p_state[0:n_max];
    n = np.arange(0,n_max,1);
    return (n,p)

def find_smallest_final_event_time(list_of_trajectories):

    def max_time(x_tuple):
        return np.max(x_tuple[0])
    result = np.min( list( map(max_time, list_of_trajectories) ) );

    return result

def interpolate_onto_same_dt( resampled_time_axis, old_time_axis, state_space):
    f = interp1d(old_time_axis, state_space, kind='previous')
    return f(resampled_time_axis)

def interpolate_all_states(x_tuple, resampled_time_axis):
    old_time = x_tuple[0];
    state_trajectory = x_tuple[1];

    number_of_states = state_trajectory.shape[1];

    state_arr = [];
    for state_index in range(number_of_states):
        specific_trajectory = state_trajectory[:,state_index]
        new_specific_trajectory = interpolate_onto_same_dt( resampled_time_axis, old_time, specific_trajectory )
        state_arr.append(new_specific_trajectory);
    return state_arr

def interpolate_all_trajectories(list_of_trajectories, timeStep):

    min_time = find_smallest_final_event_time(list_of_trajectories);
    resampled_time_axis = np.arange(start=0, stop=min_time, step=timeStep);

    resampled_trajectories = []

    for trajectory_index in range(len(list_of_trajectories)):
        resampled_trajectories.append( resample_to_fixed_step(list_of_trajectories[trajectory_index][0], list_of_trajectories[trajectory_index][1], timeStep) )

    return resampled_time_axis, resampled_trajectories

def resample_to_fixed_step(time_in, traj, timeStep ):
    num_of_states = traj.shape[1];
    time = 0;
    tFinal = time_in[-1]
    iFinal = math.ceil(tFinal/timeStep);
    time_arr = np.zeros(iFinal);
    state_arr = np.zeros([iFinal,num_of_states]);

    time_arr[0] = time;
    state_arr[0, :] = traj[0,:];

    for i in range(1,iFinal,1):
        time = time + timeStep
        time_arr[i] = time
        last_event_time_index = np.max(np.argwhere(time_in < time));
        for state in range(0,num_of_states,1):
            state_arr[i, state] = traj[last_event_time_index, state]
    return (time_arr, state_arr)