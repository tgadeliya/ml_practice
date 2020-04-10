import numpy as np
import idx2numpy


def load_data(file:str, path="data"):
    """Load data from files without 1,0 classes"""
    X_file, y_file = f"{path}/{file}-images-idx3-ubyte", f"{path}/{file}-labels-idx1-ubyte"
    
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
        return (data - self.mean)/self.std
    