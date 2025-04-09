import numpy as np
from np2p._numba_functions import gammaln_jit, gammaln_jit_vect

small_positive = 1e-10

def recursive_grid(bounds, n_pts):
    """
    Recursively generates the n-dimensional grid points (extremes are excluded).
    
    Arguments:
        list-of-lists bounds: extremes for each dimension (excluded)
        int n_pts:            number of points for each dimension
        
    Returns:
        np.ndarray: grid
    """
    bounds = np.atleast_2d(bounds)
    n_pts  = np.atleast_1d(n_pts)
    if len(bounds) == 1:
        d = np.linspace(*bounds[0], n_pts[0])
        return np.atleast_2d(d).T
    else:
        grid_nm1 = recursive_grid(np.array(bounds)[1:], n_pts[1:])
        d        = np.linspace(*bounds[0], n_pts[0])
        grid     = []
        for di in d:
            for gi in grid_nm1:
                grid.append([di,*gi])
        return np.array(grid)

def log_likelihood(x, DP):
    """
    Warning: sign inverted for optimisation
    """
    return -(_log_likelihood(x, DP) + DP.log_prior(x[:-1]))

def _log_likelihood(x, DP):
    # Base distribution
    B  = DP.model(DP.current_bins, *x[:-1])*DP.eval_selection_function
    if not all(B > 0):
        B[B==0] = small_positive
    B /= np.sum(B)
    a  = np.exp(x[-1])*B.flatten()
    # Normalisation constant
    lognorm = gammaln_jit(np.exp(x[-1])) - np.sum(gammaln_jit_vect(a))
    # Likelihood
    logL    = np.sum(np.multiply(a-1., DP.log_q[DP.current_q])) + lognorm
    return logL
