import numpy as np
import idx2numpy
import itertools

class Scaler:
    
    def fit(self, data) -> None:
        """Calculate mean, variance, min, max over columns. """
        
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        
        # Prevent division by zero
        # It is possible to do, because our variable  distribute in [0,255]
        self.std[self.std == 0.] = 1 
    
    def standardize(self, data) -> np.ndarray:
        return (data - self.mean) / self.std


def load_data(file: str) -> (np.ndarray, np.ndarray):
    """
       Load data from files without 1,0 classes.
       Use from model directory root.
       
       file argument:
         "train" - train data
         "t10k" - test data
    """
    
    X_filepath, y_filepath = f"data/{file}-images-idx3-ubyte", f"data/{file}-labels-idx1-ubyte"
    y = idx2numpy.convert_from_file(y_filepath).astype("int")
    # Create mask with appropriate classes
    data_mask = y > 1 
    X = idx2numpy.convert_from_file(X_filepath).astype("int")
    
    return X[data_mask], y[data_mask] 


def PCA(X: np.ndarray, k: int) -> np.ndarray:
    """Calcute PCA of X and choose k first eigenvectors with highest eigenvalues"""
    
    x = X.copy()
    x -= np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    eigval, eigvect = np.linalg.eig(cov)

    desc_idx_eig_val = np.argsort(eigval)[::-1]
    eig_k = eigvect[:, desc_idx_eig_val][:, :k]
    
    return x.dot(eig_k)
    

def Polynomial(X: np.ndarray) -> np.ndarray:
    """ Create second degree polynomial features from X.
        No zero degree polynomial, because bias term will be added during training.
        New matrix contains original features and all interactions between features without square of the same feature. 
    """
    
    # Convert array to Fortran order memory to speed up matrix multiplication
    x = np.asfortranarray(X)
    N, d = x.shape
    
    idx_combinations = list(itertools.combinations(range(d), 2))
    new_d = len(idx_combinations) + d # binom(N, 2) + initial features
    
    # C order for future matrix multiplication during training/evaluation phase
    X_poly = np.ones((N, new_d), order='C')
    # First column for bias
    X_poly[:, :d] = x.copy()
    
    for idx, (i, j) in enumerate(idx_combinations, d):
        X_poly[:, idx] = x[:, i] * x[:, j]
    
    return X_poly
