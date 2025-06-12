import numpy as np
from .utils import lag_matrix
from .basis import natural_cubic_spline

class CrossBasis:
    def __init__(self, x, max_lag, var_knots, lag_knots, cen_value=None):
        """
        x: exposure time series
        max_lag: max lag to consider
        var_knots: knots for variable basis (temperature, etc.)
        lag_knots: knots for lag basis
        cen_value: centering value
        """
        self.x = np.asarray(x)
        self.max_lag = max_lag
        self.cen_value = cen_value if cen_value is not None else np.mean(x)
        self.var_basis = natural_cubic_spline(x, var_knots)
        self.lag_basis = natural_cubic_spline(np.arange(max_lag + 1), lag_knots)
        self.cross_basis = self._construct_crossbasis()

    def _construct_crossbasis(self):
        lagged_x = lag_matrix(self.x, self.max_lag)  # shape: [n, L+1]
        n, Lp1 = lagged_x.shape
        var_dim = self.var_basis.shape[1]
        lag_dim = self.lag_basis.shape[1]
        cb = np.zeros((n, var_dim * lag_dim))

        # construct cross-basis via tensor product
        for t in range(n):
            vb = natural_cubic_spline(lagged_x[t, :], knots=self.var_basis_knots())
            cb_row = np.outer(vb.mean(axis=0), self.lag_basis.mean(axis=0)).flatten()
            cb[t, :] = cb_row

        return cb

    def var_basis_knots(self):
        # Needed for internal calls to spline, especially if centering is applied
        return np.linspace(np.min(self.x), np.max(self.x), self.var_basis.shape[1] - 2)

