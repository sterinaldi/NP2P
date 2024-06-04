import numpy as np
from tqdm import tqdm
from scipy.stats import poisson, multivariate_normal as mn
from scipy.special import logsumexp
from emcee import EnsembleSampler
from emcee.autocorr import AutocorrError
from parest._numba_functions import logsumexp_jit, gammaln_jit, gammaln_jit_vect

def evaluate_logP(x, self):
    return self.log_post(x)

class DirichletProcess:
    """
    Class to do parameter estimation using a set of non-parametric reconstructions.
    
    Arguments:
        callable model: parametric model
        list-of-str pars: parameters of the model
        np.ndarray bounds: 2D list of bounds, one per parameter: [[xmin, xmax], [ymin, ymax], ...]
    """
    def __init__(self, model,
                       pars,
                       bounds,
                       draws,
                       domain_bounds = None,
                       log_prior = None,
                       n_bins = None,
                       n_data = None,
                       max_a = 1e5,
                       selection_function = None,
                       burnin = 10000,
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
        if log_prior is None:
            self.log_prior = lambda x: -self.log_V
        else:
            if not callable(selection_function):
                raise Exception('log_prior must be callable')
            self.log_prior = log_prior
        # Sampler
        self.initial_state_proposer = mn(np.mean(self.bounds, axis = 1), np.identity(len(self.bounds))*np.diff(self.bounds).flatten()/50)
        self.sampler = EnsembleSampler(nwalkers = 2*(self.n_pars+1),
                                       ndim = len(self.names),
                                       log_prob_fn = evaluate_logP,
                                       args = ([self]),
                                       )
        print('Initialising MCMC')
        self.sampler.run_mcmc(initial_state = self.initial_state_proposer.rvs(2*(self.n_pars+1)),
                              nsteps        = self.burnin,
                              progress      = True,
                              )
        try:
            self.n_steps = int(np.max(self.sampler.get_autocorr_time()))
        except AutocorrError:
            print('Not thermalised yet, keep on exploring')
            self.sampler.run_mcmc(initial_state = None,
                                  nsteps        = 3*self.burnin,
                                  progress      = True,
                                  )
            self.n_steps = int(np.max(self.sampler.get_autocorr_time(quiet = True)))
    
    def log_post(self, x):
        logP = self.log_prior_full(x)
        if np.isfinite(logP):
            return logP + self.log_likelihood(x, self.N)
        return -np.inf

    def log_prior_full(self, x):
        if np.all([self.bounds[i][0] < x[i] < self.bounds[i][1] for i in range(self.n_pars+1)]):
            return self.log_prior(x[:-1])
        else:
            return -np.inf
    
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
        m  = self.model(vals, *x[:self.n_pars])*(vals[1]-vals[0])
        if not all(m > 0):
            return -np.inf
        m /= np.sum(m)
        a  = x[-1]*m
        # Normalisation constant
        lognorm = gammaln_jit(np.sum(a)) - np.sum(gammaln_jit_vect(a))
        # Likelihood
        logL    = logsumexp_jit(np.sum(np.multiply(a-1., draws), axis = 1)) - np.log(len(draws))
        return logL + lognorm
    
    def initialise(self):
        self.samples      = np.empty((0, self.n_pars_total))
        self.logP         = np.empty((0))
        self.dict_vals    = {}
        self.dict_draws   = {}
        self.dict_selfunc = {}

    def run(self, size = 1):
        size           = int(size)
        samples        = np.empty((size, self.n_pars+1))
        logP           = np.zeros(size)
        idx            = np.random.randint(2*(self.n_pars+1), size = size)
        if self.poisson is None:
            N_b = np.ones(size)*self.n_bins
        else:
            N_b    = self.poisson.rvs(size = size)
            logP_N = self.poisson.logpmf(N_b)
        for i in tqdm(range(size), desc = 'Sampling'):
            self.N = N_b[i]
            self.sampler.run_mcmc(initial_state = self.initial_state_proposer.rvs(2*(self.n_pars+1)), nsteps = 3*self.n_steps)
            samples[i] = self.sampler.get_last_sample()[0][idx[i]]
            logP[i]    = self.sampler.compute_log_prob(self.sampler.get_last_sample()[0])[0][idx[i]]
        if self.poisson is not None:
            samples  = np.hstack((samples, np.atleast_2d(N_b).T))
            logP[i] += logP_N[i]
        self.samples = np.concatenate((self.samples, samples))
        self.logP    = np.concatenate((self.logP, logP))
