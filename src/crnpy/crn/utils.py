import numpy as np
import csv
from scipy.integrate import solve_ivp
from typing import Sequence

def open_csv(file):
    # Function to open .csv files and extract them for plotting
    out_data = []
    with open(file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in data:
            out_data.append([float(datapoint) for datapoint in row[0].split(',')])
    out_data = np.asarray(out_data) 
    return out_data

def save_crn(filename, txt):
    # Function to save CRN.txt files
    f = open(filename, "w")
    f.write(txt)
    f.close()

def read_crn_txt(filename):
    # Function to parse CRN.txt files to use in python 
    f = open(filename, "r")
    species_and_nothing = {''}
    reaction_rates = []
    reaction_reacts = []
    reaction_prods = []
    initial_concs = {}
    for x in f:
        if x.startswith("#"):
            conc_string = x.replace(' ', '').strip(' \n\t#')
            split_concs = conc_string.split(",")
            for conc in split_concs:
                species_name, conc_val = conc.split("=")
                if species_name in initial_concs:
                    conc_val_stored = initial_concs[species_name]
                    max_conc = max(conc_val_stored, float(conc_val))
                    initial_concs[species_name]= float(max_conc)
                else:
                    initial_concs.update({species_name: float(conc_val)})
        else:
            x = x.replace(' ', '').strip(' \n\t')
            split_comma = x.split(",")
            rate = float(split_comma[1].strip(' \n\t'))
            split_arrow = split_comma[0].split("->")
            reactants = split_arrow[0].split("+")
            products = split_arrow[1].split("+")

            reaction_rates = reaction_rates + [rate]

            for reactant in reactants:
                species_and_nothing.add(reactant)

            for product in products:
                species_and_nothing.add(product)

            reaction_reacts = reaction_reacts + [reactants]
            reaction_prods = reaction_prods + [products]
    
    number_species = len(species_and_nothing) - 1 
    species = list(species_and_nothing)[1:]
    number_reactions = len(reaction_rates)
    
    react_stoch = np.zeros(shape=(number_reactions, number_species))
    prod_stoch = np.zeros(shape=(number_reactions, number_species))

    for r in np.arange(0, number_reactions):
        reacts_for_react = reaction_reacts[r]
        prods_for_react = reaction_prods[r]
        for s in np.arange(0, number_species):
            react_stoch[r,s] = reacts_for_react.count(species[s])
            prod_stoch[r,s] = prods_for_react.count(species[s])

    stoch_mat = prod_stoch - react_stoch
    
    initial_concs_vec = np.zeros(shape=(number_species,))
    for species_idx in np.arange(0,len(species)):
        if species[species_idx] in initial_concs:
            initial_concs_vec[species_idx] = initial_concs[species[species_idx]]

    _species = np.asarray(species)
    s_idx = _species.argsort()
    _species = list(_species[s_idx])
    _react_stoch = react_stoch[:, s_idx]
    _prod_stoch = prod_stoch[:, s_idx]
    _stoch_mat = stoch_mat[:, s_idx]
    _initial_concs_vec = initial_concs_vec[s_idx]

    return (_species, reaction_rates, _react_stoch, _prod_stoch, _stoch_mat, number_species, number_reactions, _initial_concs_vec )

def stoch_mat_to_mass_action(t, x, reaction_rates, react_stoch, stoch_mat):
    # Function that converts a stoichiometry matrix into a reaction-rate equation 
    # reaction_rates - NumPy array of reaction rates (num_of_reactions, )
    # react_stoch - NumPy array of positive ints (num_of_reactions, num_of_species )
    # stoch_mat - NumPy array of ints (num_of_reactions, num_of_species )
    # t - time value. Ignored as mass-action kinetics produce autonomous ODEs. 
    # x - current concentrations of CRN. 
    conc_to_power_of_react = np.power(x, react_stoch);
    fluxes = np.prod(conc_to_power_of_react, axis=1)
    fluxes_with_rates = np.asarray(reaction_rates)*fluxes
    mass_action = np.matmul(np.transpose(stoch_mat), fluxes_with_rates)
    return mass_action

def new_initial_conditions(old_inits, species, dict):
    new_inits = []
    id_dict = {}
    for s in np.arange(0, len(species), 1):
        if species[s] in dict.keys():
            new_inits += [dict[species[s]]]
        else:
            new_inits += [old_inits[s]]
        id_dict[species[s]] = s
    return new_inits, id_dict

def simulate_trajectory(crn_file, t_length, t_step, init_dict={}):
    # crn_file - string to .txt file in the CRN format
    # t_length - float for time period of trajectory
    # init_dict - a dict that overwrties the initial concentrations {'X_1':2, 'X_2': 2}

    # reads the previously saved chemical reaction network file returning the key stoichiometry and kinetic matrices 
    species, reaction_rates, react_stoch, prod_stoch, stoch_mat, number_species, number_reactions, initial_concs = read_crn_txt(crn_file)

    # groups stoichiometry matrices for use in ODE simulation 
    args_crn = (reaction_rates, react_stoch, stoch_mat,)

    # helper functions new_initial_conditions and stoch_mat_to_mass_action are used to simulate general CRNs
    if len(init_dict.keys()) > 0:
        initial_concs, id_dict = new_initial_conditions(initial_concs, species, init_dict)

    # time points to view the trajectory
    _t_eval =  np.arange(0, t_length, t_step)

    sol_crn = solve_ivp(stoch_mat_to_mass_action, [0, t_length], initial_concs, args=args_crn, t_eval=_t_eval, rtol=10e-8)

    return sol_crn

def convert_arrays_to_crn_text(species: Sequence[str], reaction_rates: Sequence[float], reaction_stoichiometry: Sequence[int], product_stoichiometry: Sequence[int], initial_concentrations: Sequence[float] ):
    
    s = '#'

    for s_id, species_str in enumerate(species):
        s+= species_str +'='+str(initial_concentrations[s_id])+','
    
    s = s[:-1]
    s += '\n'

    for r_id, rate in enumerate(reaction_rates):
    # Build reactant string
        reactants_list = []
        product_list = []
        for s_id, species_str in enumerate(species):
            stoichiometry_react = reaction_stoichiometry[r_id, s_id]
            stoichiometry_product = product_stoichiometry[r_id, s_id]
            for _ in range(stoichiometry_react):
                reactants_list.append(species_str)
            for _ in range(stoichiometry_product):
                product_list.append(species_str)
        
        reactants_str = " + ".join(reactants_list) if reactants_list else "0"
        product_str = " + ".join(product_list) if product_list else "0"

        s+= reactants_str + '->'+product_str+','+str(rate) +'\n'

    return s
