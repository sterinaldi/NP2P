import cpnest.model
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import gamma, dirichlet, beta
import numpy as np
from numba.extending import get_cython_function_address
from numba import vectorize, njit, jit, prange
from numpy.random import randint, shuffle
from random import sample, shuffle
import matplotlib.pyplot as plt
import ctypes
from multiprocessing import Pool
from itertools import product
from time import perf_counter
from compute_draws import random_paths
import pickle
import ray

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@jit
def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_sub(x, y): return x + np.log1p(-np.exp(y-x))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype='float64')
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    yield arr.reshape(la, -1).T

def findsubsets(l):
    return np.array([np.log(i) for i in cartesian_product(l) if np.abs(np.sum(i)-1)<1e-4])

def preprocess(N_max, samples, x_min, x_max, nthreads = 1, N_draws = 100, sample_choice = 'shuffle'):
    print('Pre-processing: subsets')
    subsets = {}
    for N in np.arange(N_max-1,N_max):
        m  = np.linspace(x_min, x_max, N)
        dm = m[1] - m[0]
        probs = []
        for samp in samples:
            p = samp(m)
            probs.append(p+np.log(dm))
        probs = np.array([p - logsumexp(p) for p in probs])
#        subset = findsubsets(np.exp(probs.T))
        subset = random_paths(np.exp(probs).T, N_draws)
        subsets[N] = subset
    return subsets

class DirichletProcess(cpnest.model.Model):
    
    def __init__(self, model, pars, bounds, samples, m_min, m_max, prior_pars = lambda x: 0, max_a = 10000, max_g = 200, max_N = 300, nthreads = 4, subsets = None, out_folder = './', load_preprocessed = True, n_draws = -1, precision = 1e-3):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.N          = len(samples)
        self.labels     = pars
        self.names      = pars + ['a', 'g']
        self.bounds     = bounds + [[1, max_a], [1e-3, max_g]]
        self.prior_pars = prior_pars
        self.m_min      = m_min
        self.m_max      = m_max
        self.model      = model
        self.precision  = precision
        self.paths      = preprocess(N_bins, samples, m_min, m_max, N_paths = N_paths, precision = self.precision)
        self.m          = np.linspace(self.m_min, self.m_max, N_bins)
        self.dm         = self.m[1] - self.m[0]
        self.K          = N_bins

    
    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            logP = -(1/x['a']) - (1/x['g'])
            pars = [x[lab] for lab in self.labels]
            logP += self.prior_pars(*pars)
        return logP
    
    def log_likelihood(self, x):
        K  = self.K
        N = self.N
        log_g  = np.log(x['g'])
        pars = [x[lab] for lab in self.labels]
        base = self.model(self.m, *pars)*self.dm
        c_par = x['a']
        log_a = np.log(c_par)
        
        g = x['g']
        a = c_par*base/base.sum()
        gammas = np.sum([numba_gammaln(ai) for ai in a])
        
        # Scritto esplicito per chiarezza
        addends = np.zeros(len(self.paths))
        for i, path in enumerate(self.path):
            deltas = np.sum(path*(a-1)) #priors are accounted for with zeros in path [p_i^(a_i-1)]
            residual = 1 - np.sum(path)
            if residual > self.precision:
                log_res  = np.log(residual)
                prior_ai = a[np.where(path == 0.)]
                DD_prior = np.sum([log_res*ai + numba_gammaln(ai) for ai in prior_ai]) - numba_gammaln(np.sum(prior_ai))
            addends[i] = deltas + log_g + DD_prior - gammas
        logL = numba_gammaln(c_par) - K*np.log(g + N) + logsumexp(addends)
        return logL
    

@njit
def numba_gammaln(x):
  return gammaln_float64(x)
