# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, binding=True, embedsignature=True
cimport cython
from cpnest.parameter cimport LivePoint
from libc.math cimport log, exp, HUGE_VAL, exp, sqrt, M_PI, fabs
from scipy.special.cython_special cimport gammaln, erf, gamma, xlogy
cimport numpy as np
import numpy as np
from numpy import random

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
# double functions

cdef double _truncated_d(double x, double a, double mmin, double mmax):
    cdef double N = (a-1)/(mmin**(1-a)-mmax**(1-a))
    if mmin < x < mmax:
        return N*x**(-a)
    else:
        return 0.
    
cdef double _normal_d(double x, double x0, double s):
    return exp(-((x-x0)**2/(2*s**2)))/(sqrt(2*M_PI)*s)

cdef double _smoothing(double x, double mmin, double d):
    return 1/(exp(d/(x - mmin) + d/(x - mmin - d) ) + 1)

cdef double _smoothing_right(double x, double mmax, double d):
    return 1/(exp(d/(mmax - x) + d/(mmax - x - d)) + 1)

########################
# Population models from O3a Pop paper

cdef np.ndarray[double,mode="c",ndim=1] _truncated(np.ndarray[double,mode="c",ndim=1] x, double a, double mmin, double mmax, double d):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double N = (a-1)/(mmin**(1-a)-mmax**(1-a))
    cdef double S = 1.
    for i in range(n):
        S = 1.
        if mmin < x[i] < mmax:
            if x[i] < mmin + d:
                S = _smoothing(x[i], mmin, d)
            res_view[i] = N*x[i]**(-a)*S
    return res

cdef np.ndarray[double,mode="c",ndim=1] _pl_peak(np.ndarray[double,mode="c",ndim=1] x, double l, double b, double mmin, double d, double mmax, double mu, double s):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double S = 1.
    for i in range(n):
        S = 1.
        if mmin < x[i] < mmax:
            if x[i] < mmin + d:
                S = _smoothing(x[i], mmin, d)
            res_view[i] = ((1-l)*_truncated_d(x[i], b, mmin, mmax) + l*_normal_d(x[i], mu, s))*S
    return res

cdef np.ndarray[double,mode="c",ndim=1] _broken_pl(np.ndarray[double,mode="c",ndim=1] x, double a1, double a2, double mmin, double mmax, double b, double d):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double mbreak = mmin + b*(mmax-mmin)
    cdef double S = 1.
    for i in range(n):
        S = 1.
        if mmin < x[i] < mmax:
            if x[i] < mmin + d:
                S = _smoothing(x[i], mmin, d)
            if x[i] < mbreak:
                res_view[i] = _truncated_d(x[i], a1, mmin, mbreak)*S/2.
            else:
                res_view[i] = _truncated_d(x[i], a2, mbreak, mmax)*S/2.
    return res

cdef np.ndarray[double,mode="c",ndim=1] _broken_pl_peak(np.ndarray[double,mode="c",ndim=1] x, double a1, double a2, double mmin, double mmax, double b, double d, double mu, double s, double l):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef np.ndarray[double,mode="c",ndim=1] pl = _broken_pl(x, a1, a2, mmin, mmax, b, d)
    cdef double[:] pl_view = pl
    for i in range(n):
        res_view[i] = (1-l)*pl_view[i] + l*_normal_d(x[i], mu, s)
    return res

cdef np.ndarray[double,mode="c",ndim=1] _multi_peak(np.ndarray[double,mode="c",ndim=1] x, double l, double lg, double b, double mmin, double d, double mmax, double mu1, double s1, double mu2, double s2):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double S = 1.
    for i in range(n):
        S = 1.
        if mmin < x[i] < mmax:
            if x[i] < mmin + d:
                S = _smoothing(x[i], mmin, d)
            res_view[i] = ((1-l)*_truncated_d(x[i], b, mmin, mmax) + l*(lg*_normal_d(x[i], mu1, s1) + (1-lg)*_normal_d(x[i], mu2, s2)))*S
    return res

def truncated(np.ndarray[double,mode="c",ndim=1] x, double a, double mmin, double mmax, double d):
    return _truncated(x, a, mmin, mmax, d)

def pl_peak(np.ndarray[double,mode="c",ndim=1] x, double l, double b, double mmin, double d, double mmax, double mu, double s):
    return _pl_peak(x, l, b, mmin, d, mmax, mu, s)

def broken_pl(np.ndarray[double,mode="c",ndim=1] x, double a1, double a2, double mmin, double mmax, double b, double d):
    return _broken_pl(x, a1, a2, mmin, mmax, b, d)

def multi_peak(np.ndarray[double,mode="c",ndim=1] x, double l, double lg, double b, double mmin, double d, double mmax, double mu1, double s1, double mu2, double s2):
    return _multi_peak(x, l, lg, b, mmin, d, mmax, mu1, s1, mu2, s2)

def broken_pl_peak(np.ndarray[double,mode="c",ndim=1] x, double a1, double a2, double mmin, double mmax, double b, double d, double mu, double s, double l):
    return _broken_pl_peak(x, a1, a2, mmin, mmax, b, d, mu, s, l)

########################
# Other models
cdef np.ndarray[double,mode="c",ndim=1] _tapered_plpeak(np.ndarray[double,mode="c",ndim=1] x, double b, double mmin, double mmax, double lmin, double lmax, double mu, double s, double l):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double N = (b-1)/(mmin**(1-b)-mmax**(1-b))
    for i in range(n):
        res_view[i] = (1-l)*N*x[i]**(-b)*(1+erf((x[i]-mmin)/(lmin)))*(1+erf((mmax-x[i])/lmax))/4. + l*_normal_d(x[i], mu, s)
    return res

def tapered_plpeak(np.ndarray[double,mode="c",ndim=1] x, double b, double mmin, double mmax, double lmin, double lmax, double mu, double s, double l):
    return _tapered_plpeak(x, b, mmin, mmax, lmin, lmax, mu, s, l)

cdef np.ndarray[double,mode="c",ndim=1] _tapered_pl(np.ndarray[double,mode="c",ndim=1] x, double b, double mmin, double mmax, double lmin, double lmax):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double app, N = 0.
    for i in range(n):
        app = x[i]**(-b)*(1+erf((x[i]-mmin)/(lmin)))*(1+erf((mmax-x[i])/(lmax)))/4
        res_view[i] = app
        N += app
    for i in range(n):
        res_view[i] = res_view[i]/N
    return res

def tapered_pl(np.ndarray[double,mode="c",ndim=1] x, double b, double mmin, double mmax, double lmin, double lmax):
    return _tapered_pl(x, b, mmin, mmax, lmin, lmax)

cdef np.ndarray[double,mode="c",ndim=1] _pl_peak_smoothed(np.ndarray[double,mode="c",ndim=1] x, double l, double b, double mmin, double d_min, double mmax, double d_max, double mu, double s):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    cdef double S = 1.
    for i in range(n):
        S = 1.
        if mmin < x[i] < mmax:
            if x[i] < mmin + d_min:
                S = _smoothing(x[i], mmin, d_min)
            if x[i] > mmax - d_max:
                S = _smoothing(mmax, x[i], d_max)
            res_view[i] = ((1-l)*_truncated_d(x[i], b, mmin, mmax) + l*_normal_d(x[i], mu, s))*S
    return res

def pl_peak_smoothed(np.ndarray[double,mode="c",ndim=1] x, double l, double b, double mmin, double d_min, double mmax, double d_max, double mu, double s):
    return _pl_peak_smoothed(x, l, b, mmin, d_min, mmax, d_max, mu, s)
    
cdef np.ndarray[double,mode="c",ndim=1] _chi2(np.ndarray[double,mode="c",ndim=1] x, double df):
    cdef unsigned int i
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[double,mode="c",ndim=1] res = np.zeros(n,dtype=np.double)
    cdef double[:] res_view = res
    
    for i in range(n):
        res_view[i] = exp(xlogy(df/2.-1, x[i]) - x[i]/2. - gammaln(df/2.) - (np.log(2.)*df)/2.)
    return res

def chi2(np.ndarray[double,mode="c",ndim=1] x, double df):
    return _chi2(x, df)
    
########################

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

def cython_log_likelihood(LivePoint LP,
                   np.ndarray[double,mode="c",ndim=1] x,
                   np.ndarray[double,mode="c",ndim=2] draws,
                   unsigned int model):

    cdef int i
    cdef int n = x.shape[0]
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
    elif model == 7:
        m = _truncated(x, LP['b'], LP['mmin'], LP['mmax'], LP['d'])*dx
    elif model == 8:
        m = _broken_pl(x, LP['a1'], LP['a2'], LP['mmin'], LP['mmax'], LP['b'], LP['d'])*dx
    elif model == 9:
        m = _pl_peak(x, LP['l'], LP['b'], LP['mmin'], LP['d'], LP['mmax'], LP['mu'], LP['s'])*dx
    elif model == 10:
        m = _multi_peak(x, LP['l'], LP['lg'], LP['b'], LP['mmin'], LP['d'], LP['mmax'], LP['mu1'], LP['s1'], LP['mu2'], LP['s2'])*dx
    elif model == 11:
        m = _broken_pl_peak(x, LP['a1'], LP['a2'], LP['mmin'], LP['mmax'], LP['b'], LP['d'], LP['mu'], LP['s'], LP['l'])*dx
    elif model == 12:
        m = _tapered_plpeak(x, LP['b'], LP['mmin'], LP['mmax'], LP['lmin'], LP['lmax'], LP['mu'], LP['s'], LP['w'])*dx
    elif model == 13:
        m = _tapered_pl(x, LP['b'], LP['mmin'], LP['mmax'], LP['lmin'], LP['lmax'])*dx
    elif model == 14:
        m = _pl_peak_smoothed(x, LP['l'], LP['b'], LP['mmin'], LP['d_min'], LP['mmax'], LP['d_max'], LP['mu'], LP['s'])*dx
    elif model == 15:
        m = _chi2(x, LP['df'])*dx
    else:
        print('model not supported, screw you!')
        return -np.inf
    cdef double concentration = LP['a']
    cdef double logL = _log_likelihood(m, draws, concentration)
    return logL
    
cdef double _log_likelihood(np.ndarray[double,mode="c",ndim=1] m,
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
    cdef double global_alpha = 0.0
    cdef double g = 0.0
    cdef double N = np.sum(m)
    for i in range(Nbins):
        a[i] = concentration*m[i]/N
        g += gammaln(a[i])
        global_alpha += a[i]
    
    cdef double lognorm = gammaln(global_alpha) - g

    for j in range(Ndraws):
        l = 0.0
        for i in range(Nbins):
            l += (a[i]-1.0)*draws[j,i]

        logL = log_add(logL,l)
    
    return logL + lognorm - log(Ndraws)

def log_likelihood(np.ndarray[double,mode="c",ndim=1] m,
                            np.ndarray[double,mode="c",ndim=1] suffstats,
                            double concentration):
    return _log_likelihood(m, suffstats, concentration)

