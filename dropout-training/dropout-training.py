import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x =  np.asarray(x, dtype=float) 
    
    if rng is None:
        rng = np.random

    keep_prob = 1 - p
    dropout_pattern = rng.binomial(1, keep_prob, size=x.shape)
    dropout_pattern = dropout_pattern / keep_prob 

    output = x * dropout_pattern

    return output, dropout_pattern