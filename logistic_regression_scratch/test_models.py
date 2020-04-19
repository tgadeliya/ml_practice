import numpy as np
from numpy.random import randint

import pytest

import torch

from src.preprocessing import load_data
from src.binary_model import LogisticRegressionBinary
from src.multiclass_model import LogisticRegressionMulti
from src.evaluation import accuracy

def is_composite(y: np.ndarray) -> np.ndarray:
    return np.where(np.isin(y, [4, 6, 8, 9]), 1, -1)

X_train, y_train = load_data("train")
X_train, y_train = X_train.reshape(-1, 28*28), y_train.reshape(-1, 1)
y_train_comp = is_composite(y_train)

@pytest.mark.parametrize("seed", randint(1000, size=2))
def test_model_random_states(seed: int):

    m1 = LogisticRegressionBinary(max_iter=1, random_state=seed)  # train one epoch
    m2 = LogisticRegressionBinary(max_iter=1, random_state=seed)

    assert np.array_equal(m2.rand_gen.randn(3, 3), m1.rand_gen.randn(3, 3))
    # More powerful generator check
    assert np.array_equal(m1.rand_gen.get_state()[1], m2.rand_gen.get_state()[1])

    m1.fit(X_train, y_train)
    m2.fit(X_train, y_train)

    assert np.array_equal(m1.w, m2.w)  # Weights check


def test_save_load():   
    m_bin = LogisticRegressionBinary(max_iter=1, random_state=23)
    m_bin.fit(X_train, y_train_comp)

    pred = m_bin.predict(X_train)
    m_bin.save_model("models/test.json")

    m2_bin = LogisticRegressionBinary(max_iter=1, random_state=13)    
    
    m2_bin.load_model("models/test.json")

    pred_loaded = m2_bin.predict(X_train)

    assert np.array_equal(m2_bin.w, m_bin.w), "Weights in loaded model differ"
    assert np.array_equal(pred, pred_loaded), "Predictions of loaded model aren't equal to initial model."


def test_multi_gradient():
    
    model = LogisticRegressionMulti(num_classes=8)
    
    weight = np.random.randn((785, 8))
    model.w = weight

    X_bias = model.add_bias(X_train[123, :])

    
    grad_model = model.calc_grad(X_bias, y_train[123])

    X_bias_t = torch.tensor(X_bias, requires_grad=True)
    weight = torch.tensor(weight, requires_grad=True)

    out = torch.nn.Softmax(dim=0)(torch.mm(X_bias_t, weight))
    loss = torch.nn.NLLLoss()(out, y_train[123])
    loss.backward()

    