import numpy as np;
import random;
from crnpy import tools;

def crn_state_rates_generator(current_state, reactant_matrix, product_matrix, reaction_rates, null_index):
    # current_state is np.array of molecule numbers for species
    # reactant_matrix is a np.array with the stoichometry values of the reactants. columns = number of species, rows = number of reations.
    # product_matrix is a np.array with the stoichometry values of the reactants. columns = number of species, rows = number of reations.
    # reaction_rates is np.array with one column that correspond to the rate of each reaction

    current_state_w_null = np.append(current_state, 0);
    stoichometry_matrix = product_matrix - reactant_matrix;

    scaled_rates = np.zeros(reaction_rates.shape)

    for i in range(reaction_rates.size):
        scaled_rates[i] = reaction_rates[i] * tools.compute_stoichiometry_terms( reactant_matrix[i,:], current_state_w_null, null_index)

    nonnegative_rates_mask = scaled_rates > 0;


    future_states = np.delete( stoichometry_matrix, null_index, 1) + current_state;
    nonnegative_states_mask = np.all(future_states >= 0, 1);


    possible_states_mask = np.logical_and(nonnegative_states_mask, nonnegative_rates_mask);

    possible_states = future_states[possible_states_mask];
    possible_rates = scaled_rates[possible_states_mask];

    return (possible_states, possible_rates)


def gillespie_simulation( tRun, start_state, reactant_matrix, product_matrix, reaction_rates, null_index ):

    t = 0;
    current_state = start_state;

    time_arr = np.array([0]);
    state_arr = np.array([start_state]);


    while t < tRun:
        total_transition_rate = 0;
        u, v = crn_state_rates_generator(current_state, reactant_matrix, product_matrix, reaction_rates, null_index);
        k = np.sum(v);
        unbiased_number_time_step = random.random();
        dt = - np.log(1 - unbiased_number_time_step ) / k;

        unbiased_number_state_change = random.random();
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
        time_arr = np.append(time_arr, t);
        state_arr = np.append(state_arr, np.array([i_new]),axis=0);

    return (time_arr, state_arr)
