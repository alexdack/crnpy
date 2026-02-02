from typing import Sequence, Optional
import numpy as np
from .utils import read_crn_txt, convert_arrays_to_crn_text
from .random import create_stoichiometry_matrices

class CRN:
    def __init__(self, species: Sequence[str], 
                 reaction_rates: Sequence[float], 
                 reaction_stoichiometry: Sequence[int], 
                 product_stoichiometry: Sequence[int],
                 *, 
                 initial_concentrations: Optional[Sequence[float]] = None):
        
        # runtime checks
        if not isinstance(species, np.ndarray):
            raise TypeError("species must be a numpy array")
        
        if not isinstance(reaction_rates,  np.ndarray):
            raise TypeError("reaction_rates must be a numpy array")       

        if not isinstance(reaction_stoichiometry,  np.ndarray):
            raise TypeError("reaction_stoichiometry must be a numpy array")    
        
        if not isinstance(product_stoichiometry,  np.ndarray):
            raise TypeError("reaction_stoichiometry must be a numpy array")   
        
        # stores as attribute 
        self.species = species
        self.reaction_rates = reaction_rates
        self.reaction_stoichiometry = reaction_stoichiometry
        self.product_stoichiometry = product_stoichiometry

        # derived attributes 
        self.number_of_species = len(species)
        self.number_of_reactions = len(reaction_rates)

        print(initial_concentrations)

        if isinstance(initial_concentrations,  np.ndarray):
            self.initial_concentrations = initial_concentrations
        
        if initial_concentrations is None:
            self.initial_concentrations = np.zeros(species.shape)

        self.stoichiometry_matrix =  product_stoichiometry - reaction_stoichiometry

    @classmethod
    def from_arrays(cls, species: Sequence[str], 
                reaction_rates: Sequence[float], 
                reaction_stoichiometry: Sequence[int], 
                product_stoichiometry: Sequence[int],
                *, 
                initial_concentrations: Optional[Sequence[float]] = None):
        return cls(species, reaction_rates, reaction_stoichiometry, product_stoichiometry, initial_concentrations =initial_concentrations)
    
    @classmethod
    def from_file(cls, filename):
        species, reaction_rates, react_stoch, prod_stoch, stoch_mat, number_species, number_reactions, initial_concs_vec = read_crn_txt(filename)
        return cls(np.asarray(species), np.asarray(reaction_rates), react_stoch, prod_stoch, initial_concentrations =initial_concs_vec)
    
    @classmethod
    def from_random(cls, number_of_species: int, number_of_reactions: int, reaction_molecularity_ratio: dict = {0: 1, 1: 1, 2: 1}, product_molecularity_ratio: dict = {0: 1, 1: 1, 2: 1}):

        species = ['S_'+ str(_+1) for _ in range(number_of_species)]
        reaction_rates = np.random.lognormal(size=(number_of_reactions,))
        initial_concs_vec = np.random.lognormal(size=(number_of_species,))

        react_stoch = create_stoichiometry_matrices(number_of_reactions, number_of_species , reaction_molecularity_ratio)
        prod_stoch = create_stoichiometry_matrices(number_of_reactions, number_of_species , product_molecularity_ratio)

        return cls(np.asarray(species), np.asarray(reaction_rates), react_stoch, prod_stoch, initial_concentrations =initial_concs_vec)

    
    def __str__(self):
        return convert_arrays_to_crn_text(self.species, self.reaction_rates, self.reaction_stoichiometry, self.product_stoichiometry, self.initial_concentrations)
    
    def __repr__(self):
        return convert_arrays_to_crn_text(self.species, self.reaction_rates, self.reaction_stoichiometry, self.product_stoichiometry, self.initial_concentrations)
