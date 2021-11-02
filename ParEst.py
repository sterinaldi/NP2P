import cpnest.model
import numpy as np
from loglikelihood import cython_log_likelihood
from scipy.special import logsumexp

class DirichletProcess(cpnest.model.Model):

    def __init__(self, model, pars, bounds, samples, x, prior_pars = lambda x: 0, max_a = 10000, nthreads = 4, out_folder = './'):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.N          = len(samples)
        self.labels     = pars
        self.names      = pars + ['a']
        self.bounds     = bounds + [[0, max_a*len(x)]]
        self.prior_pars = prior_pars
        self.model      = model
        self.draws      = samples
        self.x          = x
        self.dx         = np.abs(self.x[1] - self.x[0])
        self.K          = len(x)

    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            logP = -(1/x['a'])
#            pars = [x[lab] for lab in self.labels]
#            logP += self.prior_pars(*pars)
        return logP
    
    def log_likelihood(self, x):
        return cython_log_likelihood(x, self.x, self.draws, self.model)
