import numpy as np
import csv

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