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
# FIXME: import cython dictionary
from parest.models import models

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def log_likelihood(LivePoint LP,
                   np.ndarray[double,mode="c",ndim=1] x,
                   np.ndarray[double,mode="c",ndim=2] draws,
                   np.ndarray[double,mode="c",ndim=1] selection_function,
                   unsigned int model,
                   unsigned int n_pars,
                   unsigned int Nbins,
                   unsigned int Ndraws):
    
    return _log_likelihood(LP, x, draws, selection_function, model, n_pars, Nbins, Ndraws)

cdef double _log_likelihood(LivePoint LP,
                   np.ndarray[double,mode="c",ndim=1] x,
                   np.ndarray[double,mode="c",ndim=2] draws,
                   np.ndarray[double,mode="c",ndim=1] selection_function,
                   unsigned int model,
                   unsigned int n_pars,
                   unsigned int Nbins,
                   unsigned int Ndraws):

    cdef int i
    cdef double dx = fabs(x[0]-x[1])
    cdef double concentration = LP['a']
    cdef double norm_model = 0.
    cdef double bin_val = 0.
    
    # FIXME: cython dictionary
    cdef np.ndarray[double,mode="c",ndim=1] m = models[model](x, *LP.values[:n_pars])
    cdef np.ndarray[double,mode="c",ndim=1] filtered_m = np.zeros(Nbins,dtype=np.double)
    cdef double[:] filtered_m_view = filtered_m
    
    for i in range(Nbins):
        bin_val = m[i]*selection_function[i]*dx
        filtered_m_view[i] = bin_val
        norm_model += bin_val
    
    cdef double logL = compute_log_likelihood(filtered_m, draws, concentration, norm_model, Nbins, Ndraws)
    return logL
    
cdef double compute_log_likelihood(np.ndarray[double,mode="c",ndim=1] m,
                                   np.ndarray[double,mode="c",ndim=2] draws,
                                   double concentration,
                                   double norm_model,
                                   unsigned int Nbins,
                                   unsigned int Ndraws):
    """
    L = \frac{1}{N}\sum_i \Gamma(a)\prod_j \frac{p_{i,j}^{a*m_j-1)}{\Gamma(a*m_j)}
    """
    cdef np.ndarray[double,mode="c",ndim=1] a = np.zeros(Nbins, dtype=np.double)
    cdef unsigned int i,j
    cdef double l
    cdef double logL = -HUGE_VAL
    cdef double g = 0.0
    
    # Base distribution and normalization constant
    for i in range(Nbins):
        a[i] = concentration*m[i]/norm_model
        g += gammaln(a[i])
    cdef double lognorm = gammaln(concentration) - g
    
    # Monte Carlo integral
    for j in range(Ndraws):
        l = 0.0
        for i in range(Nbins):
            l += (a[i]-1.0)*draws[j,i]
        logL = log_add(logL,l)

    return logL + lognorm - log(Ndraws)

