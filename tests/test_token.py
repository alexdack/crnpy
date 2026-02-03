from crnpy.crn import token
import numpy as np

def test_all_reaction_tuples():
    vocab = token.all_reaction_tuples(1, 1)
    print(vocab)
    assert vocab == [((), ()), ((), (0,)), ((0,), ()), ((0,), (0,))]
    vocab = token.all_reaction_tuples(2, 2)
    print(vocab)
    assert vocab == [((), ()), ((), (0,)), ((), (1,)), ((), (0, 0)), ((), (0, 1)), ((), (1, 1)), ((0,), ()), ((0,), (0,)), ((0,), (1,)), ((0,), (0, 0)), ((0,), (0, 1)), ((0,), (1, 1)), ((1,), ()), ((1,), (0,)), ((1,), (1,)), ((1,), (0, 0)), ((1,), (0, 1)), ((1,), (1, 1)), ((0, 0), ()), ((0, 0), (0,)), ((0, 0), (1,)), ((0, 0), (0, 0)), ((0, 0), (0, 1)), ((0, 0), (1, 1)), ((0, 1), ()), ((0, 1), (0,)), ((0, 1), (1,)), ((0, 1), (0, 0)), ((0, 1), (0, 1)), ((0, 1), (1, 1)), ((1, 1), ()), ((1, 1), (0,)), ((1, 1), (1,)), ((1, 1), (0, 0)), ((1, 1), (0, 1)), ((1, 1), (1, 1))]

def test_create_vocab():
    vocab, inv_vocab = token.create_vocab(1, 1)
    print(vocab)
    print(inv_vocab)
    assert vocab[((),())] == 0
    assert inv_vocab[0] == ((),())
    assert inv_vocab[vocab[((),())]] == ((),())
    for _ in range(len(vocab.keys())):
        assert vocab[inv_vocab[_]]== _

def test_parse_matrices_into_tuples():
    reaction_stoichiometry = np.array([[1,0,1], [0,1,1]])
    product_stoichiometry = np.array([[0,0,2], [0,0,1]])
    number_of_reactions = 2

    res = token.parse_matrices_into_tuples(reaction_stoichiometry, product_stoichiometry, number_of_reactions)
    print(res)
    assert res == [((0, 2), (2, 2)), ((1, 2), (2,))]

    vocab, inv_vocab = token.create_vocab(3, 2)

    assert vocab[res[0]] == 69
    assert inv_vocab[vocab[res[0]]] == ((0, 2), (2, 2))
    assert vocab[res[1]] == 83
    assert inv_vocab[vocab[res[1]]] == ((1, 2), (2,))

def test_parse_tuples_into_matrix():
    crn_tokens = [0, 1, 2, 0]
    inv_vocab = {0: ((), ()), 1: ((0, 0,), (1,)), 2: ((), (0,))}
    number_of_reactions = len(crn_tokens)
    number_of_species = 2
    react_stoich, product_stoich = token.parse_tuples_into_matrix(crn_tokens, inv_vocab, number_of_reactions, number_of_species)
    print(react_stoich)
    print(product_stoich)
    np.testing.assert_array_equal(react_stoich, np.array([[0, 0], [2, 0], [0, 0], [0, 0]]))
    np.testing.assert_array_equal(product_stoich, np.array([[0, 0], [0, 1], [1, 0], [0, 0]]))

