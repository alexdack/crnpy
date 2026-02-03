import numpy as np
import itertools
from .random import iterate_all_molecularity_tuples
from typing import Sequence

def all_reaction_tuples(number_of_species:int, molecularity:int):
    all_combs = iterate_all_molecularity_tuples(number_of_species, molecularity)
    flatten_combs = list(itertools.chain.from_iterable(all_combs.values()))
    return list(itertools.product(flatten_combs, flatten_combs))

def create_vocab(number_of_species:int, molecularity:int):
    reactions = all_reaction_tuples(number_of_species, molecularity)
    vocab = {}
    inv_vocab = {}

    for i,r in enumerate(reactions):
        vocab[r] = i
        inv_vocab[i] = r
    
    return (vocab, inv_vocab)

def parse_matrices_into_tuples(reaction_stoichiometry: Sequence[int], product_stoichiometry: Sequence[int], number_of_reactions: int):
    
    crn_tokens = []

    for r_id in range(number_of_reactions):
        reactant_tuple = ()
        product_tuple = ()

        for s_id, s_int in enumerate(reaction_stoichiometry[r_id, :]):
            for _ in range(s_int):
                reactant_tuple += (s_id,)
        
        for p_id, p_int in enumerate(product_stoichiometry[r_id, :]):
            for _ in range(p_int):
                product_tuple += (p_id,)
        
        crn_tokens += [(reactant_tuple, product_tuple)]
    
    return crn_tokens

def parse_tuples_into_matrix(crn_tokens: Sequence[tuple], inv_vocab: dict, number_of_reactions: int, number_of_species: int):
    
    reaction_stoichiometry = np.zeros((number_of_reactions, number_of_species))
    product_stoichiometry = np.zeros((number_of_reactions, number_of_species))

    crn_tuples = [inv_vocab[_] for _ in crn_tokens]

    for i, reaction_tuple in enumerate(crn_tuples):
        reactant_tuple = reaction_tuple[0]
        product_tuple = reaction_tuple[1]

        for _ in reactant_tuple:
            reaction_stoichiometry[i, _] += 1

        for _ in product_tuple:
            product_stoichiometry[i, _] += 1
    
    return (reaction_stoichiometry.astype(np.int32), product_stoichiometry.astype(np.int32))







