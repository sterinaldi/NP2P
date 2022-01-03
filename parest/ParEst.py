import cpnest.model
import numpy as np

from parest.loglikelihood import log_likelihood
from parest.models import model_names
from scipy.special import logsumexp
from numba import jit

# Default uniform prior
@jit
def unif(*args):
    return 0

class DirichletProcess(cpnest.model.Model):

    def __init__(self, model, pars, bounds, samples, x, prior_pars = unif, max_a = 10000, out_folder = './', n_resamps = None, selection_function = None, shuffle = True):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.n_pars     = len(pars)
        self.names      = pars + ['a']
        self.bounds     = bounds + [[0, max_a*len(x)]]
        self.prior_pars = prior_pars
        self.model      = model
        self.x          = x
        self.n_bins     = len(x)
        
        self.check_model()
        
        if n_resamps is None:
            self.n_resamps = len(samples)
        else:
            self.n_resamps = n_resamps
        
        if shuffle:
            self.draws = self.shuffle_samples()
        else:
            self.draws = self.generate_resamps()
        
        if selection_function is None:
            self.selection_function = np.ones(len(x))
        else:
            if not len(selection_function) == len(x):
                print('Selection function does not have the same lenght as x')
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

    def generate_resamps(self):
    
        draws  = []
        bin_idx = np.arange(len(self.x))
        
        for _ in range(self.n_resamps):
            sample_idx = np.random.randint(0, len(self.samples), size = len(self.x))
            draw       = np.array([self.samples[si, bi] for (si, bi) in zip(sample_idx, bin_idx)])
            draw       = draw - logsumexp(draw)
            draws.append(draw)

        return np.atleast_2d(draws)
    
    def shuffle_samples(self):
    
        draws = np.copy(self.samples)
        bin   = [np.random.shuffle(d) for d in draws.T]
        
        for i in range(len(draws)):
            draws[i] = draws[i] - logsumexp(draws[i])
        
        return draws
    
    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        
        if np.isfinite(logP):
            logP = -(1/x['a'])
            pars = x.values[:-1]
            logP += self.prior_pars(*pars)
        
        return logP
    
    def log_likelihood(self, x):
        return log_likelihood(x, self.x, self.draws, self.selection_function, self.model, self.n_pars, self.n_bins, self.n_resamps)
