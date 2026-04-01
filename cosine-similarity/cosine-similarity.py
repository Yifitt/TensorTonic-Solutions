import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a,float)
    b = np.asarray(b,float)
    
    if np.linalg.norm(a)*np.linalg.norm(b) == 0:
        return 0.0
    else:
        return a@b/(np.linalg.norm(a)*np.linalg.norm(b))