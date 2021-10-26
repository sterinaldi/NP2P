import cpnest.model
import numpy as np
from loglikelihood import log_likelihood

def preprocess(N_bins, samples, x_min, x_max):
    x  = np.linspace(x_min, x_max, N_bins)
    dx = x[1] - x[0]
    probs = []
    for samp in samples:
        p = samp(x)
        probs.append(p+np.log(dx))
    probs = np.array([p - logsumexp(p) for p in probs])
    return probs

class DirichletProcess(cpnest.model.Model):

    def __init__(self, model, pars, bounds, samples, x_min, x_max, prior_pars = lambda x: 0, max_a = 10000, N_bins = 300, nthreads = 4, out_folder = './'):
    
        super(DirichletProcess, self).__init__()
        self.samples    = samples
        self.N          = len(samples)
        self.labels     = pars
        self.names      = pars + ['a']
        self.bounds     = bounds + [[1, max_a]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.model      = model
        self.draws      = preprocess(N_bins, samples, m_min, m_max)
        self.x          = np.linspace(self.m_min, self.m_max, N_bins)
        self.dx         = self.x[1] - self.x[0]
        self.K          = N_bins

    
    def log_prior(self, x):
    
        logP = super(DirichletProcess,self).log_prior(x)
        if np.isfinite(logP):
            logP = -(1/x['a'])
            pars = [x[lab] for lab in self.labels]
#            logP += self.prior_pars(*pars)
        return logP
    
    def log_likelihood(self, x):
        return log_likelihood(x, self.x, self.draws, self.model)
