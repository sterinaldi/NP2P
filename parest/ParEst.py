import numpy as np
from scipy.stats import poisson
from emcee import EnsembleSampler
from parest._numba_functions import logsumexp_jit, gammaln_jit, gammaln_jit_vect, poisson_jit, ones_jit, uniform_jit

class DirichletProcess:
    """
    Class to do parameter estimation using a set of non-parametric reconstructions.
    
    Arguments:
        callable model: parametric model
        list-of-str pars: parameters of the model
        np.ndarray bounds: 2D list of bounds, one per parameter: [[xmin, xmax], [ymin, ymax], ...]
        n_samples:
    """
    def __init__(self, model,
                       pars,
                       bounds,
                       n_samples,
                       draws,
                       domain_bounds = None
                       n_samples = 1000,
                       n_steps = 1000,
                       log_prior = None,
                       n_bins = None,
                       n_data = None,
                       max_a = 1e5,
                       out_folder = './',
                       selection_function = None,
                       ):
        self.n_pars     = len(pars)
        self.model      = model
        self.draws      = draws
        # Bins
        if domain_bounds is not None:
            self.domain_bounds = np.array(domain_bounds)
        else:
            try:
                # If FIGARO, use its bounds
                self.domain_bounds = self.draws[0].bounds[0]
            except AttributeError:
                raise Exception('Please provide domain bounds')
        if n_bins is not None:
            self.n_bins = float(n_bins)
        elif n_data is not None:
            self.exp_n_bins = np.floor(np.sqrt(n_data))
            self.poisson    = poisson(self.exp_n_bins)
        else:
            raise Exception('Please provide either n_data or n_bins')
        # Dictionaries to store pre-computed values
        self.dict_vals    = {}
        self.dict_draws   = {}
        self.dict_selfunc = {}
        # Sampler settings
        self.log_prior = log_prior
        self.names     = pars + ['a']
        self.bounds    = bounds + [[0, max_a]]
        self.log_V     = np.log(np.sum(np.diff(bounds)))
        self.n_samples = int(n_samples)
        self.n_steps   = int(n_steps)
        # Functions
        if selection_function is None:
            self.selection_function = ones_jit
        else:
            if not callable(selection_function):
                raise Exception('Selection function must be callable')
            self.selection_function = selection_function
        if log_prior is None:
            self.log_prior = uniform_jit
        else:
            if not callable(selection_function):
                raise Exception('log_prior must be callable')
            self.log_prior = log_prior
        # Sampler
        self.sampler = EnsembleSampler(nwalkers = 1,
                                       ndim = len(self.names),
                                       
                                       )
    
    def log_post(self, x):
        return self.log_prior(x) + self.log_likelihood(x, self.N)

    def log_prior_full(self, x):
        if np.isfinite(logP):
            logP  = -x[-1]
            pars  = x[:-1]
            logP += log_prior_full(*pars)
        return logP
    
    def log_likelihood(self, x, N):
        # Draws
        if not N in self.dict_draws.keys():
            vals    = np.linspace(*self.domain_bounds, N)
            selfunc = self.selection_function(vals)
            draws   = np.array([d.logpdf(vals) + np.log(vals[1]-vals[0]) for d in self.draws]).T
            draws   = np.array([np.random.choice(b, size = len(b), replace = True) for b in draws]).T
            draws   = np.array([d - logsumexp_jit(d) for d in draws])
            self.dict_draws[N]   = draws
            self.dict_vals[N]    = vals
            self.dict_selfunc[N] = selfunc
        else:
            draws   = self.dict_draws[N]
            vals    = self.dict_vals[N]
            selfunc = self.dict_selfunc[N]
        # Base distribution
        m  = self.model(vals, *x.values[:self.n_pars])*(vals[1]-vals[0])
        m /= np.sum(m)
        a  = x['a']*m
        # Normalisation constant
        lognorm = gammaln_jit(np.sum(a)) - np.sum(gammaln_jit_vect(a)) - gammaln(N)
        # Likelihood
        logL    = logsumexp_jit([np.sum(np.multiply(a-1., d)) for d in draws]) - np.log(len(draws))
        return logL + lognorm
