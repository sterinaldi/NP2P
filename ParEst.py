import cpnest.model
import numpy as np

from loglikelihood import cython_log_likelihood, normal, tapered_pl
from scipy.special import logsumexp
from numba import jit

# Default uniform prior
@jit
def unif(*args):
    return 0

class DirichletProcess(cpnest.model.Model):

    def __init__(self, model, pars, bounds, samples, x, prior_pars = unif, max_a = 10000, nthreads = 4, out_folder = './', n_resamps = None):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.labels     = pars
        self.names      = pars + ['a']
        self.bounds     = bounds + [[0, max_a*len(x)]]
        self.prior_pars = prior_pars
        self.model      = model
        self.x          = x
        
        if n_resamps is None:
            self.n_resamps = len(samples)
        else:
            self.n_resamps = n_resamps
        
        self.draws = self.generate_resamps()

    def generate_resamps(self):
    
        draws  = []
        bin_idx = np.arange(len(self.x))
        
        for _ in range(self.n_resamps):
            sample_idx = np.random.randint(0, len(self.samples), size = len(self.x))
            draw       = np.array([self.samples[si, bi] for (si, bi) in zip(sample_idx, bin_idx)])
            draw       = draw - logsumexp(draw)
            draws.append(draw)

        return np.atleast_2d(draws)

    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        
        if np.isfinite(logP):
            logP = -(1/x['a'])
            pars = x.values[:-1]
            logP += self.prior_pars(*pars)
        
        return logP
    
    def log_likelihood(self, x):
        return cython_log_likelihood(x, self.x, self.draws, self.model)
