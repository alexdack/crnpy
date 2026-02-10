from typing import Sequence, Optional
import numpy as np
from .utils import read_crn_txt, convert_arrays_to_crn_text, save_crn, simulate_trajectory
from .random import create_stoichiometry_matrices
from .token import parse_matrices_into_tuples, parse_tuples_into_matrix

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

        if isinstance(initial_concentrations,  np.ndarray):
            self.initial_concentrations = initial_concentrations
        
        if initial_concentrations is None:
            self.initial_concentrations = np.zeros(species.shape)

        self.stoichiometry_matrix =  product_stoichiometry - reaction_stoichiometry

        self.__hash__()

        # creates a dict lookup for O(1) finiding the species index
        species_lookup = {}
        for i, s in enumerate(self.species):
            species_lookup[s] = i

        self.species_lookup = species_lookup

    @classmethod
    def from_arrays(cls, species: Sequence[str], 
                reaction_rates: Sequence[float], 
                reaction_stoichiometry: Sequence[int], 
                product_stoichiometry: Sequence[int],
                *, 
                initial_concentrations: Optional[Sequence[float]] = None):
        return cls(species, reaction_rates, reaction_stoichiometry, product_stoichiometry, initial_concentrations =initial_concentrations)
    
    @classmethod
    def from_file(cls, filename: str):
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

    @classmethod
    def from_tokens(cls, stoichiometry_tokens, rate_tokens, initial_concentrations_tokens, inv_vocab):
        number_of_species = len(initial_concentrations_tokens)
        number_of_reactions = len(rate_tokens)
        species = ['S_'+ str(_+1) for _ in range(len(initial_concentrations_tokens))]
        reaction_stoichiometry, product_stoichiometry = parse_tuples_into_matrix(stoichiometry_tokens, inv_vocab, number_of_reactions, number_of_species)

        return cls(np.asarray(species), rate_tokens, reaction_stoichiometry, product_stoichiometry, initial_concentrations =initial_concentrations_tokens)

    @classmethod
    def from_dict(cls, d):
        return cls(
            species=np.asarray(d["species"]),
            reaction_rates=np.asarray(d["reaction_rates"]),
            reaction_stoichiometry=np.asarray(d["reaction_stoichiometry"]),
            product_stoichiometry=np.asarray(d["product_stoichiometry"]),
            initial_concentrations=np.asarray(d["initial_concentrations"])
        )
    
    def to_dict(self):
        return {
            "species": self.species.tolist(),
            "reaction_rates": self.reaction_rates.tolist(),
            "reaction_stoichiometry": self.reaction_stoichiometry.tolist(),
            "product_stoichiometry": self.product_stoichiometry.tolist(),
            "initial_concentrations": self.initial_concentrations.tolist(),
        }
    
    def __str__(self):
        return convert_arrays_to_crn_text(self.species, self.reaction_rates, self.reaction_stoichiometry, self.product_stoichiometry, self.initial_concentrations)
    
    def __repr__(self):
        return convert_arrays_to_crn_text(self.species, self.reaction_rates, self.reaction_stoichiometry, self.product_stoichiometry, self.initial_concentrations)
    
    def __hash__(self):
        key_str =  str(hash(self.reaction_stoichiometry.data.tobytes())) + str(hash(self.product_stoichiometry.data.tobytes())) + str(hash(self.species.data.tobytes())) + str(hash(self.reaction_rates.data.tobytes())) + str(hash(self.initial_concentrations.data.tobytes()))
        self.crn_id  = hash(key_str)
        return self.crn_id 

    def save(self, filename: str=None):
        if filename is None:
            filename = str('crn_') + str(self.crn_id) + '.txt'
        save_crn(filename, str(self))

    def integrate(self, t_length, t_step):
        return simulate_trajectory( self.reaction_rates, self.reaction_stoichiometry, self.stoichiometry_matrix, self.initial_concentrations, t_length, t_step)
    
    def distance_from(self, other_crn, species_to_compare, t_length, t_step):

        if not set(species_to_compare).issubset(set(self.species)):
            raise ValueError('species_to_compare are not present in crn object A.')
        
        if not set(species_to_compare).issubset(set(other_crn.species)):
            raise ValueError('species_to_compare are not present in crn object B.')
    
        sol_self = self.integrate(t_length, t_step)
        sol_other = other_crn.integrate(t_length, t_step)

        self_idx = np.asarray([ self.species_lookup[s_to_compare] for s_to_compare in species_to_compare ])
        other_idx = np.asarray([ other_crn.species_lookup[s_to_compare] for s_to_compare in species_to_compare ])

        self_traj = sol_self.y[self_idx, :]
        other_traj = sol_other.y[other_idx, :]

        return np.mean(np.square(self_traj - other_traj))  

    def tokenize(self, vocab, max_number_of_species, max_number_of_reaction):
        list_of_reaction_tuples = parse_matrices_into_tuples(self.reaction_stoichiometry, self.product_stoichiometry, self.number_of_reactions)
        stoichiometry_tokens = [ vocab[_] for _ in list_of_reaction_tuples ]
        rate_tokens = self.reaction_rates
        initial_concentrations_tokens = self.initial_concentrations

        if self.number_of_species < max_number_of_species:
            for _ in range(max_number_of_species - self.number_of_species):
                initial_concentrations_tokens = np.append(initial_concentrations_tokens, 0)

        if self.number_of_reactions < max_number_of_reaction:
            for _ in range(max_number_of_reaction - self.number_of_reactions):
                stoichiometry_tokens+= [0]
                rate_tokens = np.append(rate_tokens, 0)

        return (stoichiometry_tokens, rate_tokens, initial_concentrations_tokens)
    

