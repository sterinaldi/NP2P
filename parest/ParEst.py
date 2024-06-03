import raynest.model
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import logsumexp, gammaln
from numba import jit

# Default uniform prior
@jit
def unif(*args):
    return 0

class DirichletProcess(raynest.model.Model):

    def __init__(self, model,
                       pars,
                       bounds,
                       samples,
                       domain_bounds,
                       draws,
                       log_prior = None,
                       n_bins = None,
                       n_data = None,
                       max_a = 1e8,
                       selection_function = None,
                       ):
    
        super(DirichletProcess, self).__init__()
        self.names         = pars + ['a']
        self.bounds        = bounds + [[0, max_a]]
        self.model         = model
        self.n_pars        = len(pars)
        self.draws         = draws
        # Bins
        if n_bins is not None:
            self.n_bins       = int(n_bins)
            self.poisson      = None
            self.n_pars_total = self.n_pars + 1
            self.N            = self.n_bins
        elif n_data is not None:
            self.n_bins       = None
            self.exp_n_bins   = int(np.sqrt(n_data))
            self.poisson      = poisson(self.exp_n_bins)
            self.n_pars_total = self.n_pars + 2
            self.N            = self.exp_n_bins
        else:
            raise Exception('Please provide either n_data or n_bins')
        # Dictionaries to store pre-computed values
        self.dict_draws    = {}
        self.dict_vals     = {}
        self.dict_selfunc  = {}
        # Domain
        if domain_bounds is not None:
            self.domain_bounds = np.array(domain_bounds)
        else:
            try:
                # If FIGARO, use its bounds
                self.domain_bounds = self.draws[0].bounds[0]
            except AttributeError:
                raise Exception('Please provide domain bounds')
        
        # Selection function
        if selection_function is None:
            self.selection_function = lambda x: np.ones(len(x))
        else:
            if not callable(selection_function):
                raise Exception('selection_function must be callable')
            self.selection_function = selection_function
        # Prior
        if log_prior is None:
            self.log_prior = lambda x: -self.log_V
        else:
            if not callable(selection_function):
                raise Exception('log_prior must be callable')
            self.log_prior = log_prior

    def log_prior(self, x):
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            pars = x.values[:-2]
            logP += self.prior_pars(pars)
        return logP
    
    def log_likelihood(self, x):
        if not self.N in self.dict_draws.keys():
            vals    = np.linspace(*self.domain_bounds, N)
            selfunc = self.selection_function(vals)
            draws   = np.array([d.logpdf(vals) + np.log(vals[1]-vals[0]) for d in self.draws]).T
            draws   = np.array([np.random.choice(b, size = len(b), replace = True) for b in draws]).T
            draws   = np.array([d - logsumexp(d) for d in draws])
            self.dict_draws[N]   = draws
            self.dict_vals[N]    = vals
            self.dict_selfunc[N] = selfunc
        else:
            draws   = self.dict_draws[N]
            vals    = self.dict_vals[N]
            selfunc = self.dict_selfunc[N]
        # Base distribution
        m  = self.model(vals, *x.values[:self.n_pars])*(vals[1]-vals[0])
        if not all(m > 0):
            return -np.inf
        m /= np.sum(m)
        a  = x['a']*m
        # Normalisation constant
        lognorm = gammaln(np.sum(a)) - np.sum(gammaln(a)) - gammaln(N)
        # Likelihood
        logL    = logsumexp([np.sum(np.multiply(a-1., d)) for d in draws]) - np.log(len(draws))
        return logL + lognorm
