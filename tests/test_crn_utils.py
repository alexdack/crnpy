from crnpy.crn import utils
import numpy as np
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"

@pytest.fixture
def csv_path():
    TEST_DIR = Path(__file__).parent
    DATA_DIR = TEST_DIR / "data"
    csv_path = DATA_DIR / "test_1.csv"
    return csv_path

def test_open_csv(csv_path):
    np.testing.assert_array_equal(utils.open_csv(csv_path), np.array([[1., 1.]]))