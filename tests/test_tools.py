import numpy as np
from crnpy.legacy.tools import compute_stoichiometry_terms_text

def test_compute_stoichiometry_terms_text():
    assert compute_stoichiometry_terms_text( np.array([1, 0, 1]), np.array(['A', 'B', '1']), 3) == '*pow(A, 1)*pow(1, 1)'