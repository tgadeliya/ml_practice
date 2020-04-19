import numpy as np
from src.binary_model import LogisticRegressionBinary


class LogisticRegressionMulti(LogisticRegressionBinary):

    def calc_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
            Calculate gradient for every instance from batch
            returns: grad_batch: [batch_size, d, C],
                     where d - num of features, C - num of classes
        """
        y_shift = y-2 # from numbers to idx 

        lin_comb = np.dot(X, self.w)
        probs = self.softmax(lin_comb)

        true_mask = range(X.shape[0]), y_shift.ravel()
        probs[true_mask] = - (1 - probs[true_mask]) # gradients for true classes 
        grad_batch = X[:, :, np.newaxis] * probs[:, np.newaxis, :]  # [batch_size, [d,1] ] * [batch_size, [1,C]] -> [batch_size, d, c]
        return grad_batch

    def loss(self, X, y) -> float:
        """Calculate mean loss on batch"""

        y_shift = y - 2 # from numbers to idx
        probs = self.softmax(np.dot(X, self.w))
        
        loss = -np.log(probs[range(X.shape[0]), y_shift.ravel()])
        regularization = 0.5 * self.C * (self.w[1:, :] ** 2).sum()

        return np.mean(loss) + regularization

    def softmax(self, X: np.ndarray) -> np.ndarray:
        x = X.copy()
        # Numerical stability
        x -= x.max(axis=1).reshape(-1, 1)
        x_exp = np.exp(x)
        return x_exp / x_exp.sum(axis=1).reshape(-1, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1) + 2  # from idx to numbers

    def predict_composite(self, X:np.ndarray) -> float:
        preds_num = self.predict(X)
        return np.where(np.isin(preds_num, [4,6,8,9]), 1,-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.dot(self.add_bias(X), self.w)
        probs = self.softmax(out)
        return probs
      
    def evaluate(self, X:np.ndarray, y_true: np.ndarray) -> float:
        """Calculate accuracy between epoch"""
        y_pred = self.predict(X[:, 1:])
        return sum(y_pred == y_true.ravel()) / len(y_true)
