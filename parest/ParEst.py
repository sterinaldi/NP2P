import raynest.model
import numpy as np
import matplotlib.pyplot as plt

from parest.loglikelihood import log_likelihood
from parest.models import model_names, models
from scipy.special import logsumexp
from numba import jit

# Default uniform prior
@jit
def unif(*args):
    return 0

class DirichletProcess(raynest.model.Model):

    def __init__(self, model, pars, bounds, samples, x, draws, n_points, prior_pars = unif, max_a = 1e8, out_folder = './', n_resamps = None, selection_function = None, tol = 1e-1):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.n_pars     = len(pars)
        self.names      = pars + ['a', 'N']
        self.bounds     = bounds + [[0, max_a], [10, len(x)]]
        self.prior_pars = prior_pars
        self.model      = model
        self.x          = x
        self.log_dx     = np.log(x[1]-x[0])
        self.n_bins     = len(x)
        self.n_points   = n_points
        self.tol        = tol
        self.draws      = draws
        self.dict_draws = {}
        
        self.check_model()
        
        if n_resamps is None:
            self.n_resamps = len(samples)
        else:
            self.n_resamps = n_resamps
        
        # Shuffling
#        self.draws = samples #np.array([np.random.choice(b, len(b), replace = False) for b in samples.T]).T
#        self.draws = np.array([s - logsumexp(s + self.log_dx) for s in self.draws])
        
        if selection_function is None:
            self.selection_function = np.ones(len(x))
        else:
            if not len(selection_function) == len(x):
                print('Selection function does not have the same length as x')
                exit()
            self.selection_function = selection_function
    
    def check_model(self):
        try:
            print('Selected model: ' + model_names[self.model])
        except:
            print('The model you selected is not implemented (yet). You may want to try one of the following:')
            for key, name in zip(model_names.keys(), model_names.values()):
                print('{0}: {1}'.format(key, name))
            exit()

    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        
        if np.isfinite(logP):
            logP = - np.log(x['N'])
            pars = x.values[:-2]
            logP += self.prior_pars(*pars)
        
        return logP
    
    def log_likelihood(self, x):
        N = 100#int(np.sqrt(len(self.samples))) # int(x['N'])
        vals = np.linspace(self.x.min(), self.x.max(), N)
        if not N in self.dict_draws.keys():
            draws = np.array([d.logpdf(vals) - np.log(np.sum(d.pdf(vals))) for d in self.draws])
#            draws = np.array([d.pdf(vals) for d in self.draws])
            self.dict_draws[N] = draws
        else:
            draws = self.dict_draws[N]
        return log_likelihood(x, vals, draws, self.selection_function, self.model, self.n_pars, N, self.n_points, self.n_resamps)
