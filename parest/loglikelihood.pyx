# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, binding=True, embedsignature=True
cimport cython
from cpnest.parameter cimport LivePoint
from libc.math cimport log, exp, HUGE_VAL, fabs
from scipy.special.cython_special cimport gammaln
cimport numpy as np
import numpy as np
from parest.models import models

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def log_likelihood(LivePoint LP,
                   np.ndarray[double,mode="c",ndim=1] x,
                   np.ndarray[double,mode="c",ndim=2] draws,
                   unsigned int model,
                   unsigned int n_pars):
    
    return _log_likelihood(LP, x, draws, model, n_pars)

cdef double _log_likelihood(LivePoint LP,
                   np.ndarray[double,mode="c",ndim=1] x,
                   np.ndarray[double,mode="c",ndim=2] draws,
                   unsigned int model,
                   unsigned int n_pars):

    cdef int i
    cdef double dx = fabs(x[0]-x[1])
    cdef double concentration = LP['a']
    
    cdef np.ndarray[double,mode="c",ndim=1] m = models[model](x, *LP.values[:n_pars])*dx
    
    cdef double logL = compute_log_likelihood(m, draws, concentration)
    return logL
    
cdef double compute_log_likelihood(np.ndarray[double,mode="c",ndim=1] m,
                            np.ndarray[double,mode="c",ndim=2] draws,
                            double concentration):
    """
    L = \frac{1}{N}\sum_i \Gamma(a)\prod_j \frac{p_{i,j}^{a*m_j-1)}{\Gamma(a*m_j)}
    """
    cdef unsigned int Nbins = draws.shape[1]
    cdef unsigned int Ndraws = draws.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] a = np.zeros(Nbins, dtype=np.double)
    
    # compute the normalisation constants
    cdef unsigned int i,j
    cdef double l
    cdef double logL = -HUGE_VAL
    cdef double g = 0.0
    cdef double N = m.sum()
    for i in range(Nbins):
        a[i] = concentration*m[i]/N
        g += gammaln(a[i])
    
    cdef double lognorm = gammaln(concentration) - g

    for j in range(Ndraws):
        l = 0.0
        for i in range(Nbins):
            l += (a[i]-1.0)*draws[j,i]

        logL = log_add(logL,l)
    
    return logL + lognorm - log(Ndraws)

