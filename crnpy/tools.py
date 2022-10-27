import numpy as np;
import matplotlib.pyplot as plt;
from scipy.integrate import solve_ivp;

def CRNToODESystem(reactant_matrix, product_matrix, rate_vector, species_names, species_vector, null_index):
    # number_reactions, number_of_species
    # reactant_matrix = (number_reactions, number_of_species)
    # product_matrix = (number_reactions, number_of_species)
    # rate_vector = (number_reactions, 1)
    # species_names = (1, number_of_species)
    # null_index is the index of the chemical species that represents nothing (e.g destruction of a species)

    number_reactions = rate_vector.size;
    number_of_species = reactant_matrix.shape[0];
    result = np.zeros(number_of_species);

    # remove the null chemical ODE
    chemical_range = np.delete(range(number_of_species), null_index)

    for chemical_index in chemical_range:
        ode_for_chemical_text = species_names[chemical_index] + 'dot = ';
        ode_function = 0;

        change_in_chemical = product_matrix[:, chemical_index] - reactant_matrix[:, chemical_index];
        reactions_that_change_chemical = np.where(change_in_chemical != 0)[0];

        for reaction_index in reactions_that_change_chemical:

            change_in_chemical_for_given_reaction = change_in_chemical[reaction_index];
            rate_of_given_reaction = rate_vector[reaction_index];
            reactant_exponents  = reactant_matrix[reaction_index, :];
            stoichiometry_terms = compute_stoichiometry_terms( reactant_exponents, species_vector, null_index);
            ode_function = ode_function + change_in_chemical_for_given_reaction*rate_of_given_reaction*stoichiometry_terms;

            stoichiometry_terms_text = compute_stoichiometry_terms_text(reactant_exponents, species_names, null_index);
            ode_for_chemical_text = ode_for_chemical_text + ' + ' + str(change_in_chemical_for_given_reaction) + '*' +str(rate_of_given_reaction) + stoichiometry_terms_text;

        result[chemical_index] = ode_function;
        print(ode_for_chemical_text);

    return result

def compute_stoichiometry_terms( exponents, species_vector, null_index):
    result = 1;
    for species_index in range(species_vector.size):
        if species_index == null_index:
            result = result;
        else:
            result = result*pow(species_vector[species_index], exponents[species_index]);
    return result;

def compute_stoichiometry_terms_text( exponents, species_names, null_index):
    result = '';
    for species_index in range(species_names.size):
        if species_index == null_index:
            result = result;
        else:
            if exponents[species_index] == 0:
                result = result;
            else:
                result = result + '*pow(' + species_names[species_index] + ', ' + str(exponents[species_index]) + ')';
    return result;

def compute_stoichiometry_terms_stochastic_propensity( exponents, species_vector, null_index):
    result = 1;
    for species_index in range(species_vector.size):
        if species_index == null_index:
            result = result;
        else:
            val = species_vector[species_index];
            ex = exponents[species_index];
            ran = np.arange(val-ex+1, val+1 , 1);
            res = np.product(ran);
            result = result*res;
    return result;