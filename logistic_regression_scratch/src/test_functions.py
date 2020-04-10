import pytest
import numpy as np
from src.preprocessing import load_data
from src.preprocessing import Scaler



# Test load_data function
@pytest.mark.parametrize("data", ["train", "test"])
def test_load_data(data:str):
    X_train, y_train = load_data("train", path="../data")

    # types
    assert type(X_train) == np.ndarray
    assert type(y_train) == np.ndarray

    #Shapes and values
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1:] == (28,28) 

    assert np.max(X_train) == 255
    assert np.min(X_train) == 0


    # Classes
    assert sum(y_train <= 1) == 0


def test_scaler():
    scaler = Scaler()
    