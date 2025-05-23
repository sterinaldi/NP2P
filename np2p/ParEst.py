import numpy as np
import warnings
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from np2p._numba_functions import logsumexp_jit
from np2p._utils import recursive_grid, log_likelihood, implemented_processes, uniform_prior, _run_single

class DirichletProcess:
    """
    Class to perform parameter estimation using a set of non-parametric reconstructions.
    
    Arguments:
        callable model:                            parametric model
        list-of-str names:                         parameters of the model
        np.ndarray bounds:                         2D list of bounds, one per parameter: [[xmin, xmax], [ymin, ymax], ...]
        list draws:                                list of non-parametric reconstructions (either instances with pdf/logpdf method, callables or np.ndarrays)
        np.ndarray domain_bounds:                  array storing the bounds of the reconstruction
        np.ndarray bins:                           array storing the value of each bin. If None, it is built using domain_bounds
        callable log_prior:                        prior on the parameters of the model. Default is uniform.
        int or np.ndarray n_bins:                  number of bins (either single value or list of values per dimension)
        int n_data:                                number of observations used to reconstruct the non-parametric distribution. Used to estimate the number of bins
        float max_alpha:                           maximum value for the alpha parameter
        callable or np.ndarray selection_function: selection function/Users/rinaldi/Documents/projects/astro_dist/lvk_population/__pycache__
        str or Path out_folder:                    output folder
        str model_name:                            name of the model, for plotting and saving
    
    Returns:
        DirichletProcess: instance of the class
    """
    def __init__(self, model,
                       draws,
                       names              = None,
                       bounds             = None,
                       domain_bounds      = None,
                       bins               = None,
                       dV                 = None,
                       log_prior          = None,
                       n_bins             = None,
                       n_data             = None,
                       max_alpha          = 1e6,
                       min_alpha          = 1e-6,
                       selection_function = None,
                       out_folder         = '.',
                       model_name         = '',
                       process            = 'dirichlet',
                       verbose            = True,
                       optimiser          = 'global',
                       n_parallel         = 1,
                       ):
        self.model   = model
        self.draws   = draws
        if not process in implemented_processes:
            raise Exception(f'The {process} process is not implemented. Please choose between one of the following: '+', '.join(implemented_processes))
        self.process = process
        self.optimiser = optimiser
        if self.process == 'dirichlet':
            if bounds is not None:
                self.bounds = np.atleast_2d(bounds + [[np.log(min_alpha), np.log(max_alpha)]]).T
            else:
                self.bounds = np.atleast_2d([[np.log(min_alpha), np.log(max_alpha)]]).T
        else:
            if bounds is not None:
                self.bounds = np.atleast_2d(bounds).T
            else:
                raise Exception('Please provide bounds for the model parameters')
        if names is not None:
            self.names = names
        else:
            self.names = []
        # Settings
        self.model_name = model_name
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok = True)
        self.verbose = verbose
        self.n_parallel = int(n_parallel)
        # Bins
        if bins is not None:
            self.fixed_bins = False
            self.bins       = bins
            if hasattr(self.bins[0], '__iter__'):
                self.bins   = [np.atleast_2d(b).T for b in self.bins]
                self.n_dims = len(self.bins[0])
                self.N = np.array([len(b) for b in self.bins])
            else:
                self.n_dims = 1
                self.N      = np.array([len(b) for b in self.bins])
        else:
            self.fixed_bins = True
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
                    self.n_data     = n_data
                else:
                    try:
                        # If FIGARO, use stored number of samples
                        self.n_bins_dim = np.ones(self.n_dims, dtype = int)*int(np.sqrt(self.draws[0].n_pts)/self.n_dims)
                        self.n_data     = self.draws[0].n_pts
                    except AttributeError:
                        raise Exception('Please provide either n_data or n_bins')
            self.bins = recursive_grid(self.domain_bounds, self.n_bins_dim)
            if self.n_dims == 1:
                self.bins = self.bins.flatten()
            self.current_bins = self.bins
            self.N            = np.ones(len(draws))*np.sum(self.n_bins_dim)
        # Selection function
        if not callable(selection_function) and selection_function is not None:
            raise Exception('selection_function must be callable')
        self.selection_function = selection_function
        if self.fixed_bins:
            if self.selection_function is not None:
                self.eval_selection_function = self.selection_function(self.bins).flatten()
            else:
                self.eval_selection_function = np.ones(np.shape(self.bins)).flatten()
        if not callable(log_prior) and log_prior is not None:
            raise Exception('log_prior must be callable or None')
        if log_prior is None:
            self.log_prior = uniform_prior
        else:
            self.log_prior = log_prior
        # Evaluate draws
        if hasattr(draws[0], 'logpdf'):
            self.log_q = np.atleast_2d([d.logpdf(self.bins) for d in draws])
        elif hasattr(draws[0], 'pdf'):
            self.log_q = np.atleast_2d([np.log(d.pdf(self.bins)) for d in draws])
        elif callable(draws[0]):
            self.log_q = np.atleast_2d([np.log(d(self.bins)) for d in draws])
        elif isinstance(draws[0], (list, np.ndarray, float)):
            if bins is None:
                raise Exception('Please provide both bins and evaluated logpdf')
            self.log_q = draws
            if not len(self.log_q[0]) == len(self.bins[0]):
                raise Exception('The number of bins and the number of evaluated points in logpdf does not match')
        self.log_q = [log_q_i - logsumexp_jit(log_q_i) for log_q_i in self.log_q]

    def run(self, verbose = None):
        if verbose is None:
            verbose = self.verbose
        if self.n_parallel == 1:
            samples = [_run_single((self, i)) for i in tqdm(range(len(self.log_q)), desc = 'Sampling', disable = not(verbose))]
        else:
            pool    = Pool(processes = self.n_parallel)
            samples = tqdm(pool.imap_unordered(_run_single, [(self, i) for i in np.arange(len(self.log_q))], chunksize = len(self.log_q)//self.n_parallel), desc = 'Sampling', disable = not(verbose), total = len(self.log_q))
        self.samples = np.array([si for si in samples if si is not None])
        # Save data
        if self.process == 'dirichlet':
            all_names = self.names + ['beta_DP']
        else:
            all_names = self.names
        np.savetxt(Path(self.out_folder, 'posterior_samples_{}.txt'.format(self.model_name)), self.samples, header = ' '.join(all_names))

