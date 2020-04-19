from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def accuracy(model, X, y) -> float:
    N = len(y)
    assert X.shape[0] == N, "Different number of samples in X and y."  
    
    return sum(model.predict(X).ravel() == y.ravel()) / N


def confusion_matrix(y_pred, y_true) -> np.ndarray:
    
    classes = np.unique(y_true)
    num_classes = len(classes)
    
    # Shift labels by min element to got indexes for matrix.
    # np.unique() returns sorted array
    shift = classes[0] 
    
    assert len(y_pred) == len(y_true), "Different length of arrays."
    assert np.array_equal(np.unique(y_true), classes), "Different classes."
    
    confusion_matrix = np.zeros((num_classes,num_classes), dtype="int")
    preds_freq = Counter(zip(y_pred, y_true))
    
    for (i,j), freq in preds_freq.items():
        confusion_matrix[i-shift, j-shift] = freq    
    
    plt.figure(figsize=(12,12))
    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix, vmin=0, vmax=70, robust= True, annot=True,
                square=True, xticklabels=classes, 
                yticklabels=classes, 
                annot_kws={"size": 11},
                cmap="summer")
    
    return confusion_matrix