import numpy as np

def lag_matrix(x, max_lag):
    """
    Construct a lag matrix for a 1D array x with lags from 0 to max_lag.
    Each row t contains [x[t], x[t-1], ..., x[t-max_lag]]
    """
    x = np.asarray(x)
    n = len(x)
    L = max_lag + 1
    mat = np.full((n, L), np.nan)

    for lag in range(L):
        mat[lag:, lag] = x[:n - lag]

    return mat

