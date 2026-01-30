from crnpy.crn import utils
import numpy as np
from pathlib import Path
import pytest

@pytest.fixture
def csv_path():
    TEST_DIR = Path(__file__).parent
    DATA_DIR = TEST_DIR / "data"
    csv_path = DATA_DIR / "test_1.csv"
    return csv_path

@pytest.fixture
def crn_to_save():
    TEST_DIR = Path(__file__).parent
    DATA_DIR = TEST_DIR / "data"
    csv_path = DATA_DIR / "crn_to_save.txt"
    return csv_path

def test_open_csv(csv_path):
    np.testing.assert_array_equal(utils.open_csv(csv_path), np.array([[1., 1.]]))

def test_save_crn(crn_to_save):
    assert True
