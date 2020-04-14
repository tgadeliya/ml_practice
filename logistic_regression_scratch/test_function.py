import pytest
import numpy as np
from numpy.random import randint
from src.preprocessing import load_data
from src.preprocessing import Scaler
from src.models import LogisticRegressionBinary


# Test load_data function
@pytest.mark.parametrize("data", ["train", "test"])
def test_load_data(data: str):
    X_train, y_train = load_data("train")

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


@pytest.mark.parametrize("seed", randint(1000, size=2))
def test_model_random_states(seed: int):
    
    m1 = LogisticRegressionBinary(max_iter=1, random_state=seed) # train one epoch
    m2 = LogisticRegressionBinary(max_iter=1, random_state=seed)

    assert np.array_equal(m2.rand_gen.randn(3, 3), m1.rand_gen.randn(3, 3))
    # More powerful generator check
    assert np.array_equal(m1.rand_gen.get_state()[1], m2.rand_gen.get_state()[1])

    X_train, y_train = load_data("train")
    X_train, y_train = X_train.reshape(-1, 28*28), y_train.reshape(-1, 1)
    
    m1.fit(X_train, y_train) 
    m2.fit(X_train, y_train)

    assert np.array_equal(m1.w, m2.w)  # Weights check


def sanity_check():
    pass
    