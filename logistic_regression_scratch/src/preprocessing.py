import numpy as np
import idx2numpy
import itertools

def load_data(file:str):
    """Load data from files without 1,0 classes
       Use from model directory root
    """
    X_file, y_file = f"data/{file}-images-idx3-ubyte", f"data/{file}-labels-idx1-ubyte"
    
    #load labels to create appropriate class mask
    y = idx2numpy.convert_from_file(y_file)
    data_mask = y > 1 

    return idx2numpy.convert_from_file(X_file)[data_mask], y[data_mask] 


class Scaler:
    
    def fit(self, data):
        "Calculate mean, variance, min, max over columns"
        self.mean = np.mean(data, axis=0)
        # Prevernt division by zero
        self.std = np.std(data, axis=0)
        
        # Prevernt division by zero
        # We can do that, because our variable  distibute in [0,255]
        self.std[self.std == 0.] = 1
        self.max = np.max(data, axis=0)
        self.min = np.min(data, axis=0)

    def standardize(self, data):
        "Standardize mean=0"
        return (data - self.mean) / self.std
    
    
def Polynomial(X):
    X = np.asfortranarray(X[:])
    N, d = X.shape
    
    idx_combinations = list(itertools.combinations(range(d), 2))
    new_d = len(idx_combinations) + d
    
    X_poly = np.ones((N, new_d))
    X_poly[:,:d] = X[:]
    for idx, (i, j) in enumerate(idx_combinations, d):
        X_poly[:, idx] = X[:, i] * X[:,j]
    
    return X_poly

def PCAimpl(x, k):
    x -= np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    eigval, eigvect = np.linalg.eig(cov)
    desc_idx_eig_val = np.argsort(eigval)[::-1]
    
    eig_k = eigvect[:, desc_idx_eig_val][:,:k]
    return x.dot(eig_k)