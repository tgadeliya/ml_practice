import numpy as np



def test_softmax():
    N = 25
    d = 14
    a = np.random.randn((N,d)).reshape(3,-1)
    softa = model_multi.softmax(a)
    assert np.array_equal(softa.sum(axis=1), np.ones((N,)))