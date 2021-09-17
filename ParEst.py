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

def findsubsets(l):
    return np.array([np.log(i) for i in product(*l) if np.abs(np.sum(i)-1)<1e-2])


class DirichletProcess(cpnest.model.Model):
    
    def __init__(self, model, pars, bounds, samples, x_min, x_max, prior_pars = lambda x: 0, max_a = 10000, max_g = 200, max_N = 300, nthreads = 4):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.n_samps    = len(samples)
        self.labels     = pars
        self.names      = pars + ['a', 'N', 'g']
        self.bounds     = bounds + [[1, max_a], [5,max_N], [1e-3, max_g]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.model      = model
        self.prior_norm = np.log(np.exp(-self.bounds[-2][0])-np.exp(-self.bounds[-2][1]))
        self.prec_probs = {}
        self.p          = Pool(nthreads)

    
    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            logP = -(1/x['a']) - (1/x['g']) #- self.prior_norm #-np.log(x['N'])  - x['g']# - self.prior_norm
            pars = [x[lab] for lab in self.labels]
            logP += self.prior_pars(*pars)
        return logP
    
    def log_likelihood(self, x):
        N  = int(x['N'])
        m  = np.linspace(self.x_min, self.x_max, N)
        dm = m[1] - m[0]
        ns = self.n_samps
        log_g  = np.log(x['g'])
        if N in self.prec_probs.keys():
            subset = self.prec_probs[N]
        else:
            probs = []
            for samp in self.samples:
                p = np.ones(N) * -np.inf
                p = samp(m)
                probs.append(p+np.log(dm))
            probs.append(np.zeros(N))
            probs = np.array([p - logsumexp(p) for p in probs])
            subset = findsubsets(np.exp(probs.T))
            self.prec_probs[N] = subset
        
        pars = [x[lab] for lab in self.labels]
        base = np.array([self.model(mi, *pars)*dm for mi in m])
        c_par = x['a']
        log_a = np.log(c_par)
        g = x['g']
        a = c_par*base/base.sum()
        gammas = np.sum([numba_gammaln(ai) for ai in a])
        addends = np.array([np.sum(ss*(a-1)) + log_g - log_a - gammas for ss in subset])
        logL = numba_gammaln(c_par) - N*np.log(g + ns) + logsumexp(addends)
        return logL
    

@njit
def numba_gammaln(x):
  return gammaln_float64(x)
