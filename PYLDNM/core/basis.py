import numpy as np
import scipy.interpolate as si

def natural_cubic_spline(x, knots, intercept=False):
    """Construct a natural cubic spline basis matrix."""
    x = np.asarray(x)
    k = np.sort(knots)
    n = len(x)
    n_knots = len(k) + 2  # boundary knots
    X = np.zeros((n, n_knots))

    # Construct the B-spline basis
    t = np.concatenate(([k[0]] * 3, k, [k[-1]] * 3))  # knot vector
    for i in range(n_knots):
        coeff = np.zeros(n_knots)
        coeff[i] = 1
        spline = si.BSpline(t, coeff, k=3)
        X[:, i] = spline(x)

    if not intercept:
        X = X[:, 1:]

    return X

