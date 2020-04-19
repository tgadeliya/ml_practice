import numpy as np
from typing import Dict

class RandomSearch:
    """ Performs search of hyperparameters among randomly generated initial hyperparameters.
    """
    
    def __init__(self, parameters_dict : Dict[str, np.array], n_samples: int, validation_split: float = 0.9, random_state :int = 25):
        """
            args:
              n_samples - Number of random samples derived from parameters_dict.
              parameters_dict - Python dictionary with model parameters. 
                                Key - parameter name string, value - np.array with values for that parameter
              validation_split - Fraction of training data. Remain data assigned to validation
              random_state - Seed for np.random.RandomState object.
        """
        self.n_samples = n_samples
        self.param_dict = parameters_dict
        self.rand_gen = np.random.RandomState(random_state)
        self.validation_split = validation_split
        
        self.score_with_params_dict = {}
        self.best_model = None
        
        
    def fit(self, model, X, y, n_model_iter = 10):
        """
            args:
                model - Class of the model to perform Randomized Search.
                n_model_iter - Number epochs to train model, before validation score is calculated.
                
            return: Best validation score among models and model parameters 
        
        """
        # index to split data train/val
        s_idx = int(X.shape[0] * self.validation_split) 
        
        for i in range(self.n_samples):
            
            params = self.get_params_sample()
            
            logreg = model(**params, num_classes = 8) #, max_iter = n_model_iter)
            logreg.fit(X[:s_idx], y[:s_idx])
            
            preds = logreg.predict(X[s_idx:])
            acc = (y[s_idx:].ravel() == preds.ravel()).sum() /len(preds)
            
            self.score_with_params_dict[acc] = params  
            
        # Choose maximum validation score over all samples.
        max_val_score = max(self.score_with_params_dict)
        
        return max_val_score, self.score_with_params_dict[max_val_score]
        
    def get_params_sample(self):
        """Generate samples from parameters dict"""
        params = {}
        
        for param_name, param_val in self.param_dict.items():
            params[param_name] = self.rand_gen.choice(param_val, size=1)[0]
        
        return params
