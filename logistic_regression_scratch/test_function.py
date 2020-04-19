import pytest
import numpy as np
from numpy.random import randn

from src.preprocessing import *
from src.evaluation import *


@pytest.mark.parametrize("data", ["train", "test"])
def test_load_data(data: str):

    X_train, y_train = load_data("train")

    assert type(X_train) == np.ndarray
    assert type(y_train) == np.ndarray

    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1:] == (28, 28) 

    assert np.max(X_train) == 255
    assert np.min(X_train) == 0

    # Check whether 1/0 class in the y array
    assert sum(y_train <= 1) == 0


def test_Scaler():
    scaler = Scaler()
      
    x = np.random.randn(10, 10)
    scaler.fit(x)
    x_scaled = scaler.standardize(x)
    
    np.testing.assert_almost_equal(np.mean(x_scaled, axis=0), np.zeros(10),
                                   err_msg="Scaled matrix doesn't have zero mean.")
    np.testing.assert_almost_equal(np.std(x_scaled, axis=0), np.ones(10),
                                   err_msg="Scaled matrix doesn't have unit standard deviation")
    

def test_polynomial():

    X = randn(10, 25)
    new_d = + 25 + 25 * 12

    X_poly = Polynomial(X)

    assert X_poly.shape == (X.shape[0], new_d)
    assert X_poly.flags["C_CONTIGUOUS"], "Memory shape in array is wrong."
    assert np.array_equal(X_poly[:, -1], (X[:, -2] * X[:, -1]))
