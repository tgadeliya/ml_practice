
import numpy as np
import json


class LogisticRegressionBinary:
    def __init__(self, C=1, learning_rate_init=0.05, alpha=0.8, weight_decay=0.9,
                 max_iter=10, batch_size=32, val_split = 0.1,verbose=False,
                 num_classes=1, random_state=25):
        """
            args:
                C - regularization term.
                learning_rate_init - Initial learning rate.
                alpha - Momentum update parameter
                weight_decay - parameter in formula used to calculate new learning rate after every epoch
                max_iter - Number of maximum epoch training.
                val_split - Fraction of data to train and 1-val_split fraction to use as stopping criteria. If 1 - no error evaluation. 
                verbose - If True print validation loss after every epoch.
                num_classes - number of classes to predict. Made for easier inheritance.
                random_state - Seed for random generator.
        """
        self.random_state = random_state  # Save seed for generator, to use in save_model()
        self.rand_gen = np.random.RandomState(self.random_state)  # set generator specific to model object

        self.C = C
        self.lr_init = learning_rate_init
        self.alpha_momentum = alpha
        self.weight_decay = weight_decay

        self.max_iter = max_iter
      
        self.val_split = val_split if val_split!=0 else None
        

        self.w = None
        self.batch_size = batch_size
        self.val_loss_hist = []
        self.num_classes = num_classes

        self.epoch = 0
        self.verbose = verbose
        
        # create dictionary with initial model parameters.
        self.param_dict = vars(self).copy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        self.validation_split = int(self.val_split * X.shape[0])
        self.y, self.y_val = np.split(y, [self.validation_split])
        self.X, self.X_val = np.split(self.add_bias(X), [self.validation_split])

        self.d = self.X.shape[1]
        self.batch_iter = np.ceil(self.X.shape[0] / self.batch_size).astype("int")

        if (self.w is None):
            # First fit calling. Weight, Momentum initialization
            self.w = self.rand_gen.randn(self.d, self.num_classes)
            self.h = np.zeros_like(self.w)

            self.lr = self.lr_init
            self.no_epoch_improv = 0

        val_loss_history = self.train()

        del self.X, self.X_val, self.y, self.y_val
        
        return val_loss_history
      
    def train(self) -> np.ndarray:
        for e in range(1, self.max_iter+1):
            batch_idx = self.rand_gen.permutation(self.X.shape[0])  # shuffle indices every epoch

            for i in range(self.batch_iter):
                batch = batch_idx[i*self.batch_size : (i+1)*self.batch_size]
                grad_batch = self.calc_grad(self.X[batch], self.y[batch])
                self.optimizer_step_on_batch(grad_batch)

            self.epoch += 1

            # Calculate validation error and check stopping criteria
            
            if self.val_split < 1:
                val_loss = self.loss(self.X_val, self.y_val)
                self.val_loss_hist.append(val_loss)
            
                if self.verbose:
                  print(f"Epoch[{e}] Val loss={val_loss}")

                if (e>2 and (self.val_loss_hist[-1] >= self.val_loss_hist[-2])):
                    self.no_epoch_improv += 1
                else:
                    self.no_epoch_improv = 0

                if (self.no_epoch_improv > 3):
                    print("Stopping criteria is met!")
                    return self.val_loss_hist

            # If stopping criteria isn't met, continue training with new lr
            self.lr_scheduler()
        
        return self.val_loss_hist
        
    def calc_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
            Calculate gradient for every instance from batch
            returns: grad_batch: [batch_size, d],
                     where d - num of features
        """

        probs = self.sigmoid(np.dot(X, self.w))
        return (probs-y)*X + self.C*self.w.reshape(1, -1)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-x))

    def optimizer_step_on_batch(self, grad_batch: np.ndarray) -> None:
        """Perform weight update with mean gradient on batch and Momentum"""
        w_grad = np.mean(grad_batch, axis=0).reshape(-1, self.num_classes)

        self.h = self.h * self.alpha_momentum + self.lr * w_grad
        self.w -= self.h

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Calculate mean loss on batch with regularization"""

        lin_comb = np.dot(X_batch, self.w)  # [N,d] X [d,] -> [N,]
        loss_batch = np.log(1+np.exp(-1*y_batch*lin_comb))
        regularization = (0.5 * self.C) * (self.w[1:] ** 2).sum()

        return np.mean(loss_batch, axis=0)[0] + regularization

    def lr_scheduler(self) -> None:
        """Update learning rate according to formula:
            new_lr = init_lr / (1+weight_decay*t),
            where t -number of epoch trained
        """
        self.lr = (1/(1+self.weight_decay*self.epoch))*self.lr_init

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.where(probs > 0.5, 1, -1).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lin_comb = np.dot(self.add_bias(X), self.w)
        return self.sigmoid(lin_comb)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> float:
        "Calculate accuracy between epoch"
        probs = self.sigmoid(np.dot(X, self.w))
        preds = np.where(probs > 0.5, 1, -1)
        acc = sum(preds == y_true) / len(y_true)
        return acc

    def save_model(self, file_path: str) -> None:
        """
           Save model initial parameters, trained weights, losses history
           args:
                file_path - path to json file with model
        """

        model_params_dict = self.param_dict.copy()
        model_params_dict["w"] = self.w.tolist()
        del model_params_dict["rand_gen"]

        json_file = json.dumps(model_params_dict)

        with open(file_path, "w") as file:
            file.write(json_file)

    def load_model(self, file_path: str) -> None:
        "Load model from json file and set all parameters from file."

        with open(file_path, "r") as f:
            model_params_dict = json.load(f)

        model_params_dict["w"] = np.asarray(model_params_dict["w"])
        self.rand_gen = np.random.RandomState(model_params_dict["random_state"])
        self.__dict__ = model_params_dict

