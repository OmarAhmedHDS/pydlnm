import numpy as np
import pandas as pd
import importlib
from scipy.interpolate import BSpline
from sklearn.preprocessing import PolynomialFeatures

def onebasis(x, fun="ns", **kwargs):
    """
    Generate a basis matrix for different functions.
    Matches R's onebasis behavior more closely.
    
    Parameters:
    x : array-like - the predictor variable (missing values allowed)
    fun : str - name of the function to be called
    **kwargs : additional arguments passed to the specific function
    
    Returns:
    pd.DataFrame with basis matrix and attributes
    """
    # Preserve original names if available
    if hasattr(x, 'name'):
        nx = x.name
    else:
        nx = None
    
    # Convert to vector but preserve dtype when possible
    x = np.asarray(x)
    
    # Calculate range ignoring NAs (like R)
    basis_range = (np.nanmin(x), np.nanmax(x))
    
    # Extract 'cen' for consistency with R
    cen = kwargs.get('cen', None)
    
    # Prepare args like R does
    args = kwargs.copy()
    args['x'] = x
    
    # Validate basis generation
    check_onebasis(fun, args, cen)
    
    # Dynamic function calling (like R's do.call)
    basis_func = get_function(fun)
    result = basis_func(**args)
    
    # Ensure 2D matrix (even for single column)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    
    # Convert to DataFrame
    result = pd.DataFrame(result)
    
    # Set column names as in R: "b1", "b2", ...
    result.columns = [f"b{i+1}" for i in range(result.shape[1])]
    
    # Set row names if original input had a name
    if nx is not None:
        result.index.name = nx
    
    # Add attributes to match R (including all original kwargs)
    result.attrs = {
        'class': ['onebasis', 'matrix'],  # Match R's class structure
        'fun': fun,
        'range': basis_range,
        'cen': cen,
        **kwargs  # Include all original arguments
    }
    
    return result

def get_function(fun_name):
    """
    Dynamically get function by name (mimics R's do.call behavior)
    """
    # Built-in functions
    function_registry = {
        'ns': ns,
        'bs': bs,
        'ps': ps,  
        'cr': cr,
        'poly': poly,
        'strata': strata,
        'thr': thr,
        'integer': integer_basis,
        'lin': lin
    }
    
    if fun_name in function_registry:
        return function_registry[fun_name]
    
    try:
        # Try to get from current global namespace
        return globals()[fun_name]
    except KeyError:
        raise ValueError(f"Function '{fun_name}' not found")

def check_onebasis(fun, args, cen):
    """
    Validation function to match R's checkonebasis
    """
    x = args.get('x')
    if x is None:
        raise ValueError("x must be provided")
    
    # Check for valid range
    if np.all(np.isnan(x)):
        raise ValueError("All values in x are missing")
    
    # Additional checks based on function type
    if fun in ['ns', 'bs', 'ps', 'cr']:
        df = args.get('df')
        knots = args.get('knots')
        if df is None and knots is None:
            raise ValueError(f"{fun} requires either 'df' or 'knots' argument")

def ns(x, df=None, knots=None, intercept=False, Boundary_knots=None, **kwargs):
    """
    Natural Splines (cubic splines with linear tails)
    Approximates R's splines::ns() function
    """
    x = np.asarray(x)
    n = len(x)
    
    # Handle boundary knots
    if Boundary_knots is None:
        xl, xr = np.nanmin(x), np.nanmax(x)
    else:
        xl, xr = Boundary_knots
    
    # Determine interior knots
    if knots is not None:
        interior_knots = np.asarray(knots)
    else:
        if df is None:
            df = 4  # Default
        # Place knots at quantiles (like R)
        if df > 0:
            probs = np.linspace(0, 1, df + 1)[1:-1]  # Exclude 0 and 1
            interior_knots = np.nanquantile(x, probs)
        else:
            interior_knots = np.array([])
    
    # Create full knot sequence
    all_knots = np.concatenate([[xl] * 4, interior_knots, [xr] * 4])
    
    # Create B-spline basis (degree 3)
    degree = 3
    basis_matrix = np.zeros((n, len(interior_knots) + 4))
    
    for i in range(n):
        if not np.isnan(x[i]):
            # Find knot interval
            for j in range(len(all_knots) - degree - 1):
                if j < basis_matrix.shape[1]:
                    basis_matrix[i, j] = _bspline_basis(x[i], all_knots, j, degree)
    
    # Apply natural spline constraints (linear beyond boundaries)
    # This is a simplified version - full implementation would match R exactly
    if not intercept:
        basis_matrix = basis_matrix[:, 1:]  # Remove intercept column
    
    return basis_matrix

def bs(x, df=None, knots=None, degree=3, intercept=False, Boundary_knots=None, **kwargs):
    """
    B-Splines basis
    Approximates R's splines::bs() function
    """
    x = np.asarray(x)
    n = len(x)
    
    # Handle boundary knots
    if Boundary_knots is None:
        xl, xr = np.nanmin(x), np.nanmax(x)
    else:
        xl, xr = Boundary_knots
    
    # Determine interior knots
    if knots is not None:
        interior_knots = np.asarray(knots)
    else:
        if df is None:
            df = 4  # Default
        # Calculate number of interior knots
        n_interior = df - degree - (1 if intercept else 0)
        if n_interior > 0:
            probs = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior_knots = np.nanquantile(x, probs)
        else:
            interior_knots = np.array([])
    
    # Create full knot sequence
    all_knots = np.concatenate([[xl] * (degree + 1), interior_knots, [xr] * (degree + 1)])
    
    # Create B-spline basis matrix
    n_basis = len(interior_knots) + degree + 1
    basis_matrix = np.zeros((n, n_basis))
    
    for i in range(n):
        if not np.isnan(x[i]):
            for j in range(n_basis):
                basis_matrix[i, j] = _bspline_basis(x[i], all_knots, j, degree)
    
    if not intercept and basis_matrix.shape[1] > 1:
        basis_matrix = basis_matrix[:, 1:]  # Remove intercept column
    
    return basis_matrix

def _bspline_basis(x, knots, i, degree):
    """
    Compute B-spline basis function using Cox-de Boor recursion
    """
    if degree == 0:
        return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    
    # Recursive formula
    left_term = 0.0
    right_term = 0.0
    
    if knots[i + degree] != knots[i]:
        left_term = (x - knots[i]) / (knots[i + degree] - knots[i]) * _bspline_basis(x, knots, i, degree - 1)
    
    if knots[i + degree + 1] != knots[i + 1]:
        right_term = (knots[i + degree + 1] - x) / (knots[i + degree + 1] - knots[i + 1]) * _bspline_basis(x, knots, i + 1, degree - 1)
    
    return left_term + right_term

def ps(x, df=None, **kwargs):
    """
    Penalized Splines (simplified implementation)
    """
    # Use B-splines as base
    return bs(x, df=df, **kwargs)

def cr(x, df=None, **kwargs):
    """
    Cubic Regression Splines (simplified - uses natural splines)
    """
    return ns(x, df=df, **kwargs)

def poly(x, degree=1, raw=False, **kwargs):
    """
    Polynomial basis
    """
    x = np.asarray(x).reshape(-1, 1)
    
    if raw:
        # Raw polynomials: x, x^2, x^3, ...
        basis = np.column_stack([x**i for i in range(1, degree + 1)])
    else:
        # Orthogonal polynomials (simplified - not exactly like R)
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        basis = poly_features.fit_transform(x)
    
    return basis

def strata(x, **kwargs):
    """
    Stratification basis (indicator variables for unique values)
    """
    x = np.asarray(x)
    unique_vals = np.unique(x[~np.isnan(x)])
    basis = np.zeros((len(x), len(unique_vals)))
    
    for i, val in enumerate(unique_vals):
        basis[:, i] = (x == val).astype(float)
    
    return basis

def thr(x, knots, **kwargs):
    """
    Threshold basis
    """
    x = np.asarray(x)
    knots = np.asarray(knots)
    basis = np.zeros((len(x), len(knots)))
    
    for i, knot in enumerate(knots):
        basis[:, i] = (x >= knot).astype(float)
    
    return basis

def integer_basis(x, **kwargs):
    """
    Integer basis (identity for integer sequences)
    """
    x = np.asarray(x)
    return x.reshape(-1, 1)

def lin(x, **kwargs):
    """
    Linear basis (just the x values)
    """
    x = np.asarray(x)
    return x.reshape(-1, 1)