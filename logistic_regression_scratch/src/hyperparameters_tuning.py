import numpy as np

class RandomSearch:
    def __init__(self, n_samples, parameters_dict, random_state=25, validation_split=0.9):
        self.n_samples = n_samples
        self.param_dict = parameters_dict
        self.rand_gen = np.random.RandomState(random_state)
        self.validation_split = validation_split
        
        self.score_with_params_dict = {}
        self.best_model = None
        
        
    def fit(self, model, X, y, n_model_iter = 10):
        # n_iterations - Number of epoch to train model
        self.param_dict["max_iter"] = [n_model_iter]
        
        s_idx = int(X.shape[0] * self.validation_split) # index to split data train/val
        
        
        for i in range(self.n_samples):
            params = self.get_params_sample()
            
            logreg = model(**params, num_classes = 8) #, max_iter = n_model_iter)
            logreg.fit(X[:s_idx], y[:s_idx])
            
            preds = logreg.predict(X[s_idx:])
            acc = (y[s_idx:].ravel() == preds.ravel()).sum() /len(preds)
            
            print("ACC", acc)
            self.score_with_params_dict[acc] = params # 
            
        max_val_score = max(self.score_with_params_dict)
        
        return max_val_score, self.score_with_params_dict[max_val_score]
        
    def get_params_sample(self):
        params = {}
        
        for param_name, param_val in self.param_dict.items():
            params[param_name] = self.rand_gen.choice(param_val, size=1)[0]
        
        return params