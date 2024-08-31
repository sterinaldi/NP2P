import numpy as np
from pathlib import Path
import raynest
import raynest.model
from np2p._numba_functions import logsumexp_jit, gammaln_jit, gammaln_jit_vect
from np2p._utils import recursive_grid

class _Model(raynest.model.Model):
    """
    Model for a Dirichlet distribution on a fixed partition of the space as in Rinaldi, Toubiana & Gair (2024).
    It can be either fixed by the user (if the reconstruction is continuous) or induced by the binned inference directly.
    Inherits from raynest.model.Model
    
    Arguments:
        np.ndarray bins: (Nbins, Ndims) array with the values for each bin
        np.ndarray log_q: (Ndraws, Nbins) array with the evaluated draws of the initial inference
        callable model: model to fit
        list-of-str names: parameter names
        
        
    """
    def __init__(self, bins,
                       log_q,
                       model,
                       names,
                       bounds,
                       min_log_alpha = np.log(1e-5),
                       max_log_alpha = np.log(1e6),
                       selection_function = None,
                       log_prior = None
                       ):
        super(_Model,self).__init__()
        self.bins  = bins
        if len(bins.shape) == 1:
            self.dV = self.bins[1]-self.bins[0]
        else:
            self.dV    = np.prod(self.bins[1,:] - self.bins[0,:])
        self.log_q = np.atleast_2d(log_q)
        self.model = model
        # Selection effects
        if selection_function is None:
            self.selection_function = np.ones(self.log_q[0].shape)
        else:
            if callable(selection_function):
                self.selection_function = selection_function(x)
            elif selection_function.shape == self.log_q[0].shape:
                self.selection_function = selection_function
            else:
                raise Exception('The selection function shape does not match the binning shape.')
        # Prior
        if callable(log_prior) or log_prior is None:
            self.log_prior_function = log_prior
        else:
            raise Exception('Please provide a callable log_prior or set it to None for uniform.')
        
        self.names  = list(names) + ['log_alpha']
        self.bounds = list(bounds) + [[min_log_alpha, max_log_alpha]]

    def log_prior(self, x):
        logP = super(_Model, self).log_prior(x)
        if np.isfinite(logP):
            if self.log_prior_function is None:
                return 0.
            else:
                return self.log_prior_function(x.values[:-1])
        else:
            return -np.inf
    
    def log_likelihood(self, x):
        # Base distribution
        B  = self.model(self.bins, *x.values[:-1])*self.selection_function*self.dV
        if not all(B > 0):
            return -np.inf
        B /= np.sum(B)
        a  = np.exp(x['log_alpha'])*B
        # Normalisation constant
        lognorm = gammaln_jit(np.exp(x['log_alpha'])) - np.sum(gammaln_jit_vect(a))
        # Likelihood
        logL    = logsumexp_jit(np.sum(np.multiply(a-1., self.log_q), axis = 1) + lognorm) - np.log(len(self.log_q))
        logL    = logsumexp_jit(np.sum(np.multiply(a-1., self.log_q), axis = 1) + lognorm) - np.log(len(self.log_q))
        return logL
    
class DirichletProcess:
    """
    Class to do parameter estimation using a set of non-parametric reconstructions.
    
    Arguments:
        callable model: parametric model
        list-of-str pars: parameters of the model
        np.ndarray bounds: 2D list of bounds, one per parameter: [[xmin, xmax], [ymin, ymax], ...]
    """
    def __init__(self, model,
                       names,
                       bounds,
                       draws,
                       domain_bounds = None,
                       bins = None,
                       log_prior = None,
                       n_bins = None,
                       n_data = None,
                       max_alpha = 1e8,
                       selection_function = None,
                       out_folder = '.',
                       verbose = False,
                       model_name = '',
                       ):
        self.model  = model
        self.draws  = draws
        self.bounds = bounds
        self.names  = names
        # Settings
        self.model_name = model_name
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok = True)
        self.verbose = int(verbose*2)
        # Bins
        if bins is not None:
            self.bins   = bins
            self.n_dims = len(bins[0])
            self.N      = len(bins)
            self.dV     = np.prod(self.bins[1,:] - self.bins[0,:])
        else:
            if domain_bounds is not None:
                self.domain_bounds = np.atleast_2d(domain_bounds)
            else:
                try:
                    # If FIGARO, use its bounds
                    self.domain_bounds = self.draws[0].bounds[0]
                except AttributeError:
                    raise Exception('Please provide domain bounds as a (Ndim, 2) array')
            self.n_dims = len(self.domain_bounds)
            if n_bins is not None:
                self.n_bins_dim = np.array(n_bins)
            else:
                if n_data is not None:
                    self.n_bins_dim = np.ones(self.n_dims, dtype = int)*int(np.sqrt(n_data)/self.n_dims)
                else:
                    try:
                        # If FIGARO, use stored number of samples
                        self.n_bins_dim = np.ones(self.n_dims, dtype = int)*int(np.sqrt(self.draws[0].n_pts)/self.n_dims)
                    except AttributeError:
                        raise Exception('Please provide either n_data or n_bins')
            self.bins = recursive_grid(self.domain_bounds, self.n_bins_dim)
            self.dV = np.prod(self.bins[1,:] - self.bins[0,:])
            if self.n_dims == 1:
                self.bins = self.bins.flatten()
            self.N    = np.sum(self.n_bins_dim)
        # Selection function
        if not callable(selection_function) and selection_function is not None:
            raise Exception('selection_function must be callable')
        self.selection_function = selection_function
        if not callable(log_prior) and log_prior is not None:
            raise Exception('log_prior must be callable or None')
        self.log_prior = log_prior
        # Evaluate draws
        if hasattr(draws[0], 'logpdf'):
            self.log_q = np.atleast_2d([d.logpdf(self.bins) + np.log(self.dV) for d in draws])
        elif hasattr(draws[0], 'pdf') or callable(draws[0]):
            self.log_q = np.atleast_2d([np.log(d.pdf(self.bins)*self.dV) for d in draws])
        elif isinstance(draws[0], (list, np.ndarray, float)):
            if bins is None:
                raise Exception('Please provide both bins and evaluated logpdf')
            self.log_q = np.atleast_2d(draws*self.dV)
            if not len(log_q[0]) == len(self.bins):
                raise Exception('The number of bins and the number of evaluated points in logpdf does not match')
#        self.log_q = np.array([np.random.choice(b, size = len(b), replace = True) for b in self.log_q.T]).T
#        self.log_q = np.array([log_q_i - logsumexp_jit(log_q_i) for log_q_i in self.log_q])
        self.log_q = np.array([self.log_q[10]])
        # Sampler
        self.DD_model = _Model(bins               = self.bins,
                               log_q              = self.log_q,
                               model              = self.model,
                               names              = self.names,
                               bounds             = self.bounds,
                               max_log_alpha      = np.log(max_alpha),
                               selection_function = self.selection_function,
                               log_prior          = self.log_prior,
                               )
        self.sampler = raynest.raynest(self.DD_model,
                                       verbose   = self.verbose,
                                       nnest     = 1,
                                       nensemble = 1,
                                       nlive     = 1000,
                                       maxmcmc   = 5000,
                                       output    = self.out_folder,
                                       )
                                       
    def run(self):
        self.sampler.run()
        # Posterior samples and evidence
        NS_samples          = self.sampler.posterior_samples.ravel()
        self.samples        = np.array([NS_samples[lab] for lab in self.DD_model.names]).T
        self.samples[:,-1]  = np.exp(self.samples[:,-1])/self.N
        self.logZ           = self.sampler.logZ
        # Save data
        np.savetxt(Path(self.out_folder, 'posterior_samples_{}.txt'.format(self.model_name)), self.samples, header = ' '.join(self.names + ['beta']))
        np.savetxt(Path(self.out_folder, 'logZ_{}.txt'.format(self.model_name)), np.atleast_1d([self.logZ]))
