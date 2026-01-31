from typing import Sequence, Optional
import numpy as np

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