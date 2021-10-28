# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, binding=True, embedsignature=True
cimport cython
from cpnest.parameter cimport LivePoint
from libc.math cimport log, exp, HUGE_VAL, exp, sqrt, M_PI, fabs
from scipy.special.cython_special cimport gammaln, erf, gamma
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

cdef np.ndarray[double,mode="c",ndim=1] _uniform(np.ndarray[double,mode="c",ndim=1] x, double x_min, double x_max):
    return np.ones(x.shape[0], dtype = np.double)/(x_max - x_min)

def uniform(np.ndarray[double,mode="c",ndim=1] x, double x_min, double x_max):
    return _uniform(x, x_min, x_max)

cdef np.ndarray[double,mode="c",ndim=1] _exponential(np.ndarray[double,mode="c",ndim=1] x, double x0, double l):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    for i in range(n):
        res_view[i] = exp(-fabs(x[i]-x0)/l)/(2*l)
    return res
    
def exponential(np.ndarray[double,mode="c",ndim=1] x, double x0, double l):
    return _exponential(x,x0,l)

cdef np.ndarray[double,mode="c",ndim=1] _cauchy(np.ndarray[double,mode="c",ndim=1] x, double x0, double g):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    for i in range(n):
        res_view[i] = g/M_PI * 1/((x[i] - x0)**2 + g**2)
    return res

def cauchy(np.ndarray[double,mode="c",ndim=1] x, double x0, double g):
    return _cauchy(x, x0, g)

cdef np.ndarray[double,mode="c",ndim=1] _generalized_normal(np.ndarray[double,mode="c",ndim=1] x, double x0, double s, double b):
    # See https://en.wikipedia.org/wiki/Generalized_normal_distribution
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    for i in range(n):
        res_view[i] = b/(2*s*gamma(1/b)) * exp(-(fabs(x[i]-x0)/s)**b)
    return res

def generalized_normal(np.ndarray[double,mode="c",ndim=1] x, double x0, double s, double b):
    return _generalized_normal(x, x0, s, b)

cdef np.ndarray[double,mode="c",ndim=1] _power_law(np.ndarray[double,mode="c",ndim=1] x,
                                                   double alpha,
                                                   double x_cut,
                                                   double l_cut):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double pre_PL = (alpha-1)/(x_cut**(1-alpha))
    cdef double N      = 1 + pre_PL*x_cut**(-alpha)*l_cut*np.sqrt(2*M_PI)/2.
    for i in range(n):
        if x[i] < x_cut:
            res_view[i] = exp(-(x[i]-x_cut)**2/(2*l_cut**2))*pre_PL*x_cut**(-alpha)/N
        else:
            res_view[i] = pre_PL*x[i]**(-alpha) / N
    return res

def power_law(np.ndarray[double,mode="c",ndim=1] x,
                                                   double alpha,
                                                   double x_cut,
                                                   double l_cut):
    return _power_law(x, alpha, x_cut, l_cut)

cdef _bimodal(np.ndarray[double,mode="c",ndim=1] x,double x0,double s0,double x1,double s1, double w):
    return w*_normal(x,x0,s0)+(1.0-w)*_normal(x,x1,s1)

def bimodal(np.ndarray[double,mode="c",ndim=1] x,double x0,double s0,double x1,double s1, double w):
    return w*_normal(x,x0,s0)+(1.0-w)*_normal(x,x1,s1)

########################

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
    elif model == 1:
        m = _power_law(x, LP['beta'], LP['xcut'], LP['lcut'])*dx
    elif model == 2:
        m = _bimodal(x, LP['m1'], LP['s1'], LP['m2'], LP['s2'], LP['w'])*dx
    elif model == 3:
        m = _uniform(x, LP['x_min'], LP['x_max'])*dx
    elif model == 4:
        m = _exponential(x, LP['x0'], LP['l'])*dx
    elif model == 5:
        m = _cauchy(x, LP['x0'], LP['g'])*dx
    elif model == 6:
        m = _generalized_normal(x, LP['x0'], LP['s'], LP['b'])*dx
    else:
        print('model not supported, screw you!')
        exit()

    cdef double concentration = LP['a']
    cdef double logL = _log_likelihood(m, draws, concentration)
    return logL
    
cdef double _log_likelihood(np.ndarray[double,mode="c",ndim=1] m,
                            np.ndarray[double,mode="c",ndim=2] draws,
                            double concentration):
    """
    L = \frac{1}{N}\sum_i \Gamma(a)\prod_j \frac{p^{a*m_j-1)}{\Gamma(a*m_j)}
    """
    cdef unsigned int Nbins = draws.shape[1]
    cdef unsigned int Ndraws = draws.shape[0]
#    cdef np.ndarray[double,mode="c",ndim=1] g = np.zeros(Nbins, dtype=np.double)
    cdef np.ndarray[double,mode="c",ndim=1] a = np.zeros(Nbins, dtype=np.double)
    
    # compute the nnormalisation constants
    cdef unsigned int i,j
    cdef double l
    cdef double logL = -HUGE_VAL
    cdef double global_alpha = 0.0
    cdef double g = 0.0
    for i in range(Nbins):
        a[i] = concentration*m[i]
        g += gammaln(a[i])
        global_alpha += a[i]
    
    cdef double lognorm = gammaln(global_alpha) - g
    
    for j in range(Ndraws):
        l = 0.0
        for i in range(Nbins):
            l += (a[i]-1.0)*draws[j,i]

        logL = log_add(logL,l + lognorm)
    
    return logL-log(Ndraws)
