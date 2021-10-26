# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, boundscheck=False, wraparound=False, binding=True, embedsignature=True

from __future__ import division
cimport numpy as np
import numpy as np
from libc.math cimport exp, sqrt, M_PI
from scipy.special.cython_special cimport erf

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
