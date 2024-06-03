import raynest.model
import numpy as np

from scipy.stats import poisson, dirichlet
from scipy.special import logsumexp, gammaln
from parest._numba_functions import gammaln_jit, gammaln_jit_vect, logsumexp_jit

class DirichletProcess(raynest.model.Model):

    def __init__(self, model,
                       pars,
                       bounds,
                       domain_bounds,
                       draws,
                       log_prior = None,
                       n_bins = None,
                       n_data = None,
                       max_a = 1e5,
                       selection_function = None,
                       ):
    
        super(DirichletProcess, self).__init__()
        self.names  = pars + ['a']
        self.bounds = bounds + [[0, 1e5]]
        self.log_V  = np.log(np.sum(np.diff(self.bounds)))
        self.model  = model
        self.n_pars = len(pars)
        self.draws  = draws
        # Bins
        if n_bins is not None:
            self.n_bins       = int(n_bins)
            self.poisson      = None
            self.n_pars_total = self.n_pars + 1
            self.N            = self.n_bins
        else:
            self.n_bins = None
            if n_data is not None:
                self.exp_n_bins = int(np.sqrt(n_data))
            else:
                try:
                    # If FIGARO, use stored number of samples
                    self.exp_n_bins = int(np.sqrt(self.draws[0].n_pts))
                except AttributeError:
                    raise Exception('Please provide either n_data or n_bins')
            self.poisson      = poisson(self.exp_n_bins)
            self.n_pars_total = self.n_pars + 2
            self.N            = self.exp_n_bins
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
            self.log_prior_pars = lambda y: -self.log_V
        else:
            if not callable(selection_function):
                raise Exception('log_prior must be callable')
            self.log_prior_pars = log_prior

    def log_prior(self, x):
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            pars  = x.values[:self.n_pars]
            return self.log_prior_pars(pars) - x['a']
        return logP
    
    def log_likelihood(self, x):
        if not self.N in self.dict_draws.keys():
            vals    = np.linspace(*self.domain_bounds, self.N)
            selfunc = self.selection_function(vals)
            draws   = np.array([d.logpdf(vals) + np.log(vals[1]-vals[0]) for d in self.draws]).T
            draws   = np.array([np.random.choice(b, size = len(b), replace = True) for b in draws]).T
            draws   = np.array([d - logsumexp(d) for d in draws])
            self.dict_draws[self.N]   = draws
            self.dict_vals[self.N]    = vals
            self.dict_selfunc[self.N] = selfunc
        else:
            draws   = self.dict_draws[self.N]
            vals    = self.dict_vals[self.N]
            selfunc = self.dict_selfunc[self.N]
        # Base distribution
        pars = x.values[:self.n_pars]
        m    = self.model(vals, *pars)*(vals[1]-vals[0])
        if not all(m > 0):
            return -np.inf
        m /= np.sum(m)
        a  = x['a']*m
        # Normalisation constant
        lognorm = gammaln_jit(np.sum(a)) - np.sum(gammaln_jit_vect(a))
        # Likelihood
        logL    = logsumexp_jit(np.sum(np.multiply(a-1., draws), axis = 1)) - np.log(len(draws))
        return logL + lognorm
