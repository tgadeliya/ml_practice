from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def accuracy(model, X, y):
    preds = model.predict(X)
    acc_score = sum(preds == y) / len(y)
    return acc_score


def confusion_matrix(y_pred, y_true):
    
    classes = np.unique(y_true)
    num_classes = len(classes)
    
    assert len(y_pred) == len(y_true), "Different length of arrays."
    assert np.array_equal(np.unique(y_true), classes), "Different classes."
    
    confusion_matrix = np.zeros((num_classes,num_classes), dtype="int")
    preds_freq = Counter(zip(y_pred, y_true))
    
    for (i,j), freq in preds_freq.items():
        confusion_matrix[i-2, j-2] = freq    
    
    plt.figure(figsize=(12,12))
    sns.set(font_scale=1.4)
    sns.heatmap(conf, vmin=0, vmax=70, robust= True, annot=True,
                square=True, xticklabels=classes, 
                yticklabels=classes, 
                annot_kws={"size": 11},
                cmap="summer")
    
    
    return confusion_matrix