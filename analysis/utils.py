import numpy as np


def get_subtract_avg_matrix(dim):
    """
    Get a matrix M such that M @ x is the same as x - avg(x).
    Which is the same as zeroing out the diagonal of x.
    """
    # z zeros out diagonal
    z = np.eye(dim) - np.ones((dim, dim)) / dim
    return z
