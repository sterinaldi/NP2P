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
from numpy cimport ndarray

#FIXME: add new models to the list (see bottom)


##############################

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

cdef np.ndarray[double,mode="c",ndim=1] _bimodal(np.ndarray[double,mode="c",ndim=1] x,double x0,double s0,double x1,double s1, double w):
    return w*_normal(x,x0,s0)+(1.0-w)*_normal(x,x1,s1)

def bimodal(np.ndarray[double,mode="c",ndim=1] x,double x0,double s0,double x1,double s1, double w):
    return w*_normal(x,x0,s0)+(1.0-w)*_normal(x,x1,s1)

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
    cdef double app, N = (b-1)/(mmin**(1-b)-mmax**(1-b))
    for i in range(n):
        res_view[i] = N*x[i]**(-b)*(1+erf((x[i]-mmin)/(lmin)))*(1+erf((mmax-x[i])/(lmax)))/4
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
    cdef double df_int = np.round(df)
    
    for i in range(n):
        res_view[i] = exp(xlogy(df_int/2.-1, x[i]) - x[i]/2. - gammaln(df_int/2.) - (np.log(2.)*df_int)/2.)
    return res

def chi2(np.ndarray[double,mode="c",ndim=1] x, double df):
    return _chi2(x, df)
    
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

########################

models = {
    0:  normal,
    1:  uniform,
    2:  exponential,
    3:  cauchy,
    4:  generalized_normal,
    5:  power_law,
    6:  bimodal,
    7:  truncated,
    8:  pl_peak,
    9:  broken_pl,
    10: multi_peak,
    11: tapered_plpeak,
    12: tapered_pl,
    13: pl_peak_smoothed,
    14: chi2,
}

model_names = {
    0:  'Gaussian',
    1:  'Uniform',
    2:  'Exponential',
    3:  'Cauchy',
    4:  'Generalized Normal',
    5:  'Power law',
    6:  'Bimodal Gaussian',
    7:  'Truncated Power Law',
    8:  'Power Law + Peak',
    9:  'Broken Power Law',
    10: 'Multi Peak',
    11: 'Tapered Power Law + Peak',
    12: 'Tapered Power Law',
    13: 'Smoothed Power Law + Peak',
    14: 'Chi^2',
}
