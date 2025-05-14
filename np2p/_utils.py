import numpy as np
import warnings
from scipy.optimize import minimize, dual_annealing
from np2p._numba_functions import gammaln_jit, gammaln_jit_vect

small_positive = 1e-200
implemented_processes = ['dirichlet', 'poisson']

def uniform_prior(x):
    return 0.

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

def _run_single(args):
    DP, i = args
    if not DP.fixed_bins:
        if DP.selection_function is None:
            selfunc = np.ones(np.shape(DP.bins[i])).flatten()
        else:
            selfunc = DP.selection_function(DP.bins[i]).flatten()
    else:
        selfunc = DP.eval_selection_function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
#        try:
        if DP.optimiser == 'local':
            sample = minimize(log_likelihood, np.mean(DP.bounds.T, axis = 1), args = (DP, i, selfunc), bounds = DP.bounds.T).x
        elif DP.optimiser == 'global':
            sample = dual_annealing(log_likelihood, args = [DP, i, selfunc], bounds = list(np.atleast_2d(DP.bounds).T)).x
#        except Exception:
#            return None
    if DP.process == 'dirichlet':
        sample[-1] = np.exp(sample[-1])/DP.N[i]
    return sample

def log_likelihood(x, DP, current_q, selfunc):
    """
    Warning: sign inverted for optimisation
    """
    if DP.process == 'dirichlet':
        log_prior = DP.log_prior(x[:-1])
    elif DP.process == 'poisson':
        log_prior = DP.log_prior(x)
    return -(_log_likelihood(x, DP, current_q, selfunc) + log_prior)

def _log_likelihood(x, DP, current_q, selfunc):
    if DP.process == 'dirichlet':
        pars = x[:-1]
    elif DP.process == 'poisson':
        pars = x
    # Base distribution
    B  = DP.model(DP.current_bins, *pars)*DP.eval_selection_function
    if not all(B > small_positive):#0):
        B[B < small_positive] = small_positive
#        B[B == 0] = small_positive
    B /= np.sum(B)
    if DP.process == 'dirichlet':
        a  = np.exp(x[-1])*B.flatten()
        lognorm = gammaln_jit(np.exp(x[-1])) - np.sum(gammaln_jit_vect(a))
        logL    = np.sum(np.multiply(a-1., DP.log_q[current_q])) + lognorm
    elif DP.process == 'poisson':
        # Counts
        exp_counts = (DP.n_data*B)
        obs_counts = (DP.n_data*np.exp(DP.log_q[current_q]))
        logL       = np.sum(obs_counts*np.log(exp_counts) - exp_counts - gammaln_jit_vect(obs_counts))
    return logL
