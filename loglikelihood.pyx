# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, binding=True, embedsignature=True
cimport cython
from cpnest.parameter cimport LivePoint
from libc.math cimport log, exp, HUGE_VAL, exp, sqrt, M_PI, fabs
from scipy.special.cython_special cimport gammaln, erf
cimport numpy as np
import numpy as np

cdef np.ndarray[double,mode="c",ndim=1] _normal(np.ndarray[double,mode="c",ndim=1] x, double x0, double s):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    for i in range(n):
        res_view[i] = exp(-((x[i]-x0)**2/(2*s**2)))/(sqrt(2*M_PI)*s)
    return res

def normal(np.ndarray[double,mode="c",ndim=1] x, double x0, double s):
    return _normal(x, x0, s)

cdef np.ndarray[double,mode="c",ndim=1] _power_law(np.ndarray[double,mode="c",ndim=1] x,
                                                   double alpha,
                                                   double x_min,
                                                   double x_max,
                                                   double l_min,
                                                   double l_max):

    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    for i in range(n):
        res_view[i] = ((1-alpha)/(x_max**(1-alpha) - x_min**(1-alpha)))*x**(-alpha)*(1+erf((x-x_min)/(l_min)))*(1+erf((x_max-x)/l_max))/4.
    return res

def power_law(np.ndarray[double,mode="c",ndim=1] x,
              double alpha,
              double x_min,
              double x_max,
              double l_min,
              double l_max):
    return _power_law(x, alpha, x_min, x_max, l_min, l_max)

def bimodal(np.ndarray[double,mode="c",ndim=1] x,double x0,double s0,double x1,double s1, double w):
    return w*_normal(x,x0,s0)+(1.0-w)*_normal(x,x1,s1)

def power_law_peak(np.ndarray[double,mode="c",ndim=1] x,
            double alpha,
            double x_min,
            double x_max,
            double l_min,
            double l_max,
            double mu,
            double s,
            double w):
    return (1-w)*_power_law(x, alpha, x_min, x_max, l_min, l_max) + w*_normal(x,mu,s)


cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def log_likelihood(LivePoint LP,
                   np.ndarray[double,mode="c",ndim=1] x,
                   np.ndarray[double,mode="c",ndim=2] draws,
                   unsigned int model):

    cdef double dx = fabs(x[0]-x[1])
    cdef np.ndarray[double,mode="c",ndim=1] m
    #FIXME: we must make sure the base distributions are normalised to start with
    if model == 0:
        m = _normal(x, LP['mean'], LP['sigma'])*dx
#    elif model == 1:
#        m = _power_law()

    else:
        print('model not supported, screw you!')
        exit()

    cdef double concentration = LP['a']
    cdef double logL = _log_likelihood(m, draws, concentration)
    return logL
    
cdef double _log_likelihood(np.ndarray[double,mode="c",ndim=1] m,
                            np.ndarray[double,mode="c",ndim=2] draws,
                            double concentration):

    cdef unsigned int Nbins = draws.shape[1]
    cdef unsigned int Ndraws = draws.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] g = np.zeros(Nbins, dtype=np.double)
    cdef np.ndarray[double,mode="c",ndim=1] a = np.zeros(Nbins, dtype=np.double)
    
    # compute the nnormalisation constants
    cdef unsigned int i,j
    cdef double lognorm = gammaln(concentration)
    cdef double l
    cdef double logL = -HUGE_VAL
    
    for i in range(Nbins):
        a[i] = concentration*m[i]
        g[i] = gammaln(a[i])
    
    for j in range(Ndraws):
        l = 0.0
        for i in range(Nbins):
            l += (a[i]-1.0)*draws[i,j]-g[i]

        logL = log_add(logL,l)

    lognorm = gammaln(concentration)
    
    return logL-log(Ndraws)+lognorm
