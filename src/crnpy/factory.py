from .crn.crn_class import CRN

def create_crn(species, reaction_rates, reaction_stoichiometry, product_stoichiometry, *, initial_concentrations = None):
    return CRN(species, reaction_rates, reaction_stoichiometry, product_stoichiometry, initial_concentrations =initial_concentrations)