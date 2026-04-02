import numpy as np

def he_initialization(W, fan_in):
    W = np.asarray(W)
    he = np.sqrt(6 / fan_in)
    W = (2 * W - 1) * he
    
    return W