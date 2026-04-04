import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """

    length = len(y_true)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum(y_true == y_pred)

    micro = 2*tp/(2*tp+((length-tp)*2))

    return micro

    