import numpy as np

class LogisticRegressionBinary:
    
    def __init__(self, C=1, 
                 learning_rate_init=0.05,
                 alpha=0.5,
                 weight_decay=0.9,
                 max_iter = 10,
                 batch_size = 32,random_state = 25,
                 tol = 0.001,
                 logging = False,
                 num_classes=1):
        
        self.rand_gen = np.random.RandomState(random_state) # set generator specific to model object
        
        self.C = C
        self.lr_init = learning_rate_init
        
        self.alpha_momentum = alpha
        self.max_iter = max_iter
        
        self.batch_size = batch_size
        
        self.tol = tol
        self.weight_decay = weight_decay
        self.w = None;
        self.loss_history = []
        self.num_classes = num_classes
        
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        
        if (self.w is None): # First fit start
            self.X = self.add_bias(X)
            self.y = y[:]
            self.N, self.d = self.X.shape
            self.batch_iter = np.ceil(self.N / self.batch_size).astype("int")
            
            self.w = self.rand_gen.randn(self.d, self.num_classes)
            self.h = np.zeros_like(self.w)
        
        self.lr = self.lr_init
        self.epoch = 0 
        self.no_epoch_improv = 0
        
        self.train()
            
    def train(self) -> None:
        for e in range(1, self.max_iter+1): # If stopping criteria isn't met, train till max_iter
            batch_idx = self.rand_gen.permutation(self.N) # shuffle indexes every epoch
            
            for i in range(self.batch_iter):
                batch = batch_idx[i*self.batch_size:(i+1)*self.batch_size]
                grad_batch = self.calc_grad(self.X[batch], self.y[batch])
                self.optimizer_step_on_batch(grad_batch)
            
            self.epoch += 1
            train_loss = self.loss(self.X, self.y)
            train_acc = self.evaluate(self.X, self.y)
            
            print(f"Epoch {e}: train_loss: {train_loss}, train_acc: {train_acc}")
            print("_____________________________")
            
            self.loss_history.append(train_loss)
            
            self.lr_scheduler()
            
            self.no_epoch_improv = (self.no_epoch_improv + 1) if (e > 2 and (self.loss_history[-1] - self.loss_history[-2])<self.tol) else 0
            
            if (self.no_epoch_improv > 3):
                print("Stopping criteria is met!")
                break
    
    def loss(self, X_batch, y_batch):
        lin_comb = np.dot(X_batch, self.w) # [N,d] X [d,] -> [N,]
        
        loss_batch = np.log(1+np.exp(-1*y_batch*lin_comb))
        regularization = (0.5 * self.C) * (self.w[1:] ** 2).sum()
        
        return np.mean(loss_batch, axis=0) + regularization
        
    def calc_grad(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        grad_batch = (self.sigmoid(np.dot(X, self.w)) - y) * X + self.C*self.w.reshape(1,-1)
        return grad_batch
    
    def lr_scheduler(self):
        self.lr = (1/(1+self.weight_decay*self.epoch))*self.lr_init 
    
    def optimizer_step_on_batch(self, grad_batch : np.ndarray) -> None:    
        w_grad = np.mean(grad_batch, axis=0).reshape(-1,self.num_classes)
        
        self.h = self.h * self.alpha_momentum + self.lr * w_grad
        self.w -= self.h 
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-1*x))
    
    def add_bias(self, X) :
        return np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:        
        probs = self.predict_proba(X)
        return np.where(probs>0.5, 1, -1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:        
        out = np.dot(self.add_bias(X), self.w)
        probs = self.sigmoid(out)
        return probs 
  
    def evaluate(self, X, y_true):
        "Calculate accuracy between epoch"
        probs = self.sigmoid(np.dot(X, self.w))
        y_pred = np.where(probs>0.5, 1, -1)
        acc = sum(y_pred == y_true)/ len(y_true)
        return acc
    
class LogisticRegressionMulti(LogisticRegressionBinary):
    
    def predict(self, X: np.ndarray) -> np.ndarray:        
        return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:        
        out = np.dot(self.add_bias(X), self.w)
        probs = self.softmax(out)
        return probs 
    
    def softmax(self, x):
        "Softmax over batch"
        x -= x.max(axis=1).reshape(-1,1)
        x_exp = np.exp(x)
        return x_exp / x_exp.sum(axis=1).reshape(-1, 1)
        
    def loss(self, X, y):
        y_shift = y - 2
        probs = self.softmax(np.dot(X, self.w))
        loss = -np.log(probs[range(X.shape[0]),y_shift.ravel()])
        regularization = (0.5 * self.C) * (self.w[1:,:] ** 2).sum()
        
        return np.mean(loss) + regularization
    
    def calc_grad(self,X:np.ndarray, y:np.ndarray) -> tuple:
        y_shift = y-2 # Shift over deleted classes
        
        lin_comb = np.dot(X, self.w)
        probs = self.softmax(lin_comb)
        
        probs[range(X.shape[0]), y_shift.ravel()] = - (1 - probs[range(X.shape[0]), y_shift.ravel()]) 
        grad_batch = X[:,:,np.newaxis] * probs[:,np.newaxis,:]   # [batch_size, [d,1] ] * [batch_size, [1,C]] -> [batch_size, d, c]         
        return grad_batch 
    
    def predict(self, X: np.ndarray) -> np.ndarray:        
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)+2
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:        
        out = np.dot(self.add_bias(X), self.w)
        probs = self.softmax(out)
        return probs 
    
    def evaluate(self, X, y_true):
        "Calculate accuracy between epoch"
        y_pred = self.predict(X[:,1:])
        acc = sum(y_pred == y_true.ravel())/ len(y_true)
        return acc