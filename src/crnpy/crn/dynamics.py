import numpy as np
from scipy.linalg import null_space

def compute_conservation_laws(stoichiometry_matrix, initial_concentrations):
    # Compute the null space of the stoichiometry matrix
    null_space_matrix = null_space(stoichiometry_matrix)
    C = null_space_matrix.T @ initial_concentrations
    return (null_space_matrix, C)

def vars_julia_solver(species):
    return " ".join(species)

def f_mass_action_julia_solver(species, reaction_rates, react_stoch, stoch_mat):
    conc_to_power_of_react = []
    for r_id in range(react_stoch.shape[0]):
      row_conc_to_power_of_react = []
      for s_id in range(react_stoch.shape[1]):
        if react_stoch[r_id, s_id] == 1:
          row_conc_to_power_of_react +=  [species[s_id]]
        elif react_stoch[r_id, s_id] > 1:
          row_conc_to_power_of_react +=  [species[s_id]+'^'+str(react_stoch[r_id, s_id])]
        else:
          row_conc_to_power_of_react +=  ['']
      conc_to_power_of_react += [row_conc_to_power_of_react]
    
    kin_mat = []
    for r_id in range(react_stoch.shape[0]):
        rate = reaction_rates[r_id]
        stoch_mat_rates = stoch_mat[r_id, :]*rate
        kin_mat += [stoch_mat_rates]

    fluxes = []
    for r_id in range(react_stoch.shape[0]):
      filt = test_list = [i for i in conc_to_power_of_react[r_id] if i != '']
      row_fluxes = '*'.join(filt)
      fluxes += [row_fluxes]

    mass_action = ['' for _ in range(react_stoch.shape[1])]

    for r_id in range(react_stoch.shape[0]):
      terms = fluxes[r_id]
      rates = kin_mat[r_id]

      for s_id in range(react_stoch.shape[1]):
        if rates[s_id] != 0.0:
          sign_val = '+' if rates[s_id] > 0 else '-'
          if terms == '':
            term_to_add =  sign_val + str(abs(rates[s_id]))
          else:
            term_to_add =  sign_val + str(abs(rates[s_id])) + '*' + terms
          mass_action[s_id] += term_to_add

    return mass_action

def f_conservation_julia_solver(species, stoch_mat, initial_concentrations):

    lhs_nullspace, C = compute_conservation_laws(stoch_mat, initial_concentrations)
    conservation_laws = []

    for i in range(lhs_nullspace.shape[1]):
        law = ' + '.join([str(lhs_nullspace[j,i]) + '*' + species[j] for j in range(lhs_nullspace.shape[0]) if lhs_nullspace[j,i] != 0]) + str(-C[i])
        conservation_laws += [law]
    return conservation_laws

def steady_states_julia_solver(species, reaction_rates, react_stoch, stoch_mat, initial_concentrations):
    mass_action = f_mass_action_julia_solver(species, reaction_rates, react_stoch, stoch_mat)
    f_cons = f_conservation_julia_solver(species, stoch_mat, initial_concentrations)
    f = mass_action + f_cons
    vars = vars_julia_solver(species)
    ode_system = ", ".join(f)
    return vars, ode_system
