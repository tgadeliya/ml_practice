import numpy as np


def accuracy(y_pred, y_true):
    return (y_pred == y_true) / len(y_true)

def train_test(m, X_train, y_train, X_test, y_test):
    
    m.fit(X_train,y_train)
    
    preds = m.predict(X_train)

    print("Train: ", accuracy_score(y_train, preds))
    
    
    preds_test = m.predict(X_test)
    print("Test: ",accuracy_score(y_test, preds_test))
    