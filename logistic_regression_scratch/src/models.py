import numpy as np

class LogisticRegressionBinary:
    
    def __init__(self, C=1, learning_rate=0.05, alpha=0.5,
                 max_iter = 10, eps = 10e-6,
                 batch_size = 32, shuffle=False):
        
        self.C = C
        self.lr = learning_rate;
        self.alpha_momentum = alpha
        
        
        self.max_iter = max_iter
        self.eps = eps
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.w = None;
        self.loss_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        
        self.X = X#self.add_bias(X)
        self.y = y[:]
        self.N, self.d = self.X.shape
        
        self.w = np.random.randn(self.d, 1)
        self.h = np.zeros((self.d,1))
        
        for e in range(1, self.max_iter+1):
            
            self.train()
            
            #val_improv = 
            #if val_improv < self.eps and (e > 1):
            #    print("Stopping criteria is met!")
            #    break
        
    def train(self) -> None:
        "Perform one epoch train"
        
        batch_idx = list(range(self.N))
        
        if self.shuffle:
            np.random.shuffle(batch_idx)
        
        batch_iter = np.ceil(self.N/self.batch_size)
        avg_loss = 0.0

        for idx in range(int(batch_iter)):
            batch = batch_idx[idx*self.batch_size:(idx+1)*self.batch_size]
            avg_batch_loss, grad_batch = self.calc_loss_and_grad(self.X[batch], self.y[batch])
            
            self.optimizer_step_on_batch(grad_batch)
            avg_loss += avg_batch_loss
        
        self.loss_history.append(avg_loss/batch_iter) # Mean example loss   
    
    def calc_loss_and_grad(self, X:np.ndarray, y:np.ndarray) -> tuple:
        
        lin_comb = np.dot(X, self.w) # [N,d] X [d,] -> [N,]
        
        loss = -np.log(1+np.exp(-1*y*lin_comb)) +  self.C * np.linalg.norm(self.w, ord=2)
        grad_batch = (self.sigmoid(lin_comb) - y) * X + self.C*2*self.w.reshape(1,-1)
        #print(f"loss :{loss}")
        #print("Gradients along rows :", grad_batch.sum(axis=1))
        
        return np.mean(loss), grad_batch
    
    
    def optimizer_step_on_batch(self, grad_batch : np.ndarray) -> None:    
        w_grad = np.mean(grad_batch, axis=0).reshape(-1,1)
        
        self.h = self.h * self.alpha_momentum + self.lr * w_grad
        self.w -= self.h 
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-1*x))
    
    def add_bias(self, X) :
        return np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:        
        out = np.dot(X, self.w)
        #out = np.dot(self.add_bias(X), self.w)
        probs = self.sigmoid(out)
        return np.where(probs>0.5, 1, -1)
    
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:        
        out = np.dot(X, self.w)
        #out = np.dot(self.add_bias(X), self.w)
        probs = self.sigmoid(out)
        return probs 
    
    
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        acc = (y_true == y_pred)
        return sum(acc)/len(y_true)


    
    
                               
class LogisticRegressionMultiClass(LogisticRegression):
    def get_loss_with_grad(self, X:np.ndarray, y:np.ndarray) -> tuple:
        pass
    
    def predict_prob(self, X):
        return np.dot(X,self.w)

    
    
class LogisticRegressionSGD(LogisticRegressionBinary):
    
    def __init__(self, C=1, learning_rate=0.05, alpha=0.5, max_iter = 10, eps = 10e-6, shuffle=False):
        
        self.C = C
        self.lr = learning_rate;
        self.alpha_momentum = alpha
        
        
        self.max_iter = max_iter
        self.eps = eps
        
        self.shuffle = shuffle
        
        self.w = None;
        self.loss_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        
        self.X = self.add_bias(X)
        self.y = y[:]
        self.N, self.d = self.X.shape
        
        self.w = np.random.randn(self.d, 1)
        self.h = np.zeros((self.d,1))
        
        for e in range(1, self.max_iter+1):
            
            self.train()
            
            #val_improv = 
            #if val_improv < self.eps and (e > 1):
            #    print("Stopping criteria is met!")
            #    break
        
    def train(self) -> None:
        "Perform one epoch train"
        
        batch_idx = list(range(self.N))
        
        if self.shuffle:
            np.random.shuffle(batch_idx)
        
        batch_iter = np.ceil(self.N/self.batch_size)
        avg_loss = 0.0

        for idx in range(int(batch_iter)):
            batch = batch_idx[idx*self.batch_size:(idx+1)*self.batch_size]
            avg_batch_loss, grad_batch = self.calc_loss_and_grad(self.X[batch], self.y[batch])
            
            self.optimizer_step_on_batch(grad_batch)
            avg_loss += avg_batch_loss
        
        self.loss_history.append(avg_loss/batch_iter) # Mean example loss   
    
    def calc_loss_and_grad(self, X:np.ndarray, y:np.ndarray) -> tuple:
        
        lin_comb = np.dot(X, self.w) # [N,d] X [d,] -> [N,]
        
        loss = -np.log(1+np.exp(-1*y*lin_comb)) +  self.C * np.linalg.norm(self.w, ord=2)
        grad_batch = (self.sigmoid(lin_comb) - y) * X + self.C*2*self.w.reshape(1,-1)
        #print(f"loss :{loss}")
        #print("Gradients along rows :", grad_batch.sum(axis=1))
        
        return np.mean(loss), grad_batch
    
    
    def optimizer_step_on_batch(self, grad_batch : np.ndarray) -> None:    
        w_grad = np.mean(grad_batch, axis=0).reshape(-1,1)
        
        self.h = self.h * self.alpha_momentum + self.lr * w_grad
        self.w -= self.h 
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-1*x))
    
    def add_bias(self, X) :
        return np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:        
        out = np.dot(X, self.w)
        #out = np.dot(self.add_bias(X), self.w)
        probs = self.sigmoid(out)
        return np.where(probs>0.5, 1, -1)
    
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:        
        out = np.dot(X, self.w)
        #out = np.dot(self.add_bias(X), self.w)
        probs = self.sigmoid(out)
        return probs 
    
    
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        acc = (y_true == y_pred)
        return sum(acc)/len(y_true)

