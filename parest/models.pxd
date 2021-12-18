cimport numpy as np
from numpy cimport ndarray
########################
cdef np.ndarray[double,mode="c",ndim=1] _normal(np.ndarray[double,mode="c",ndim=1] x, double x0, double s)
cdef np.ndarray[double,mode="c",ndim=1] _uniform(np.ndarray[double,mode="c",ndim=1] x, double x_min, double x_max)
cdef np.ndarray[double,mode="c",ndim=1] _exponential(np.ndarray[double,mode="c",ndim=1] x, double x0, double l)
cdef np.ndarray[double,mode="c",ndim=1] _cauchy(np.ndarray[double,mode="c",ndim=1] x, double x0, double g)
cdef np.ndarray[double,mode="c",ndim=1] _generalized_normal(np.ndarray[double,mode="c",ndim=1] x, double x0, double s, double b)
cdef np.ndarray[double,mode="c",ndim=1] _power_law(np.ndarray[double,mode="c",ndim=1] x, double alpha, double x_cut, double l_cut)
cdef np.ndarray[double,mode="c",ndim=1] _bimodal(np.ndarray[double,mode="c",ndim=1] x,double x0,double s0,double x1,double s1, double w)
cdef np.ndarray[double,mode="c",ndim=1] _truncated(np.ndarray[double,mode="c",ndim=1] x, double a, double mmin, double mmax, double d)
cdef np.ndarray[double,mode="c",ndim=1] _pl_peak(np.ndarray[double,mode="c",ndim=1] x, double l, double b, double mmin, double d, double mmax, double mu, double s)
cdef np.ndarray[double,mode="c",ndim=1] _broken_pl(np.ndarray[double,mode="c",ndim=1] x, double a1, double a2, double mmin, double mmax, double b, double d)
cdef np.ndarray[double,mode="c",ndim=1] _broken_pl_peak(np.ndarray[double,mode="c",ndim=1] x, double a1, double a2, double mmin, double mmax, double b, double d, double mu, double s, double l)
cdef np.ndarray[double,mode="c",ndim=1] _multi_peak(np.ndarray[double,mode="c",ndim=1] x, double l, double lg, double b, double mmin, double d, double mmax, double mu1, double s1, double mu2, double s2)
cdef np.ndarray[double,mode="c",ndim=1] _tapered_plpeak(np.ndarray[double,mode="c",ndim=1] x, double b, double mmin, double mmax, double lmin, double lmax, double mu, double s, double l)
cdef np.ndarray[double,mode="c",ndim=1] _tapered_pl(np.ndarray[double,mode="c",ndim=1] x, double b, double mmin, double mmax, double lmin, double lmax)
cdef np.ndarray[double,mode="c",ndim=1] _pl_peak_smoothed(np.ndarray[double,mode="c",ndim=1] x, double l, double b, double mmin, double d_min, double mmax, double d_max, double mu, double s)
cdef np.ndarray[double,mode="c",ndim=1] _chi2(np.ndarray[double,mode="c",ndim=1] x, double df)
########################
cdef double _truncated_d(double x, double a, double mmin, double mmax)
cdef double _normal_d(double x, double x0, double s)
cdef double _smoothing(double x, double mmin, double d)
########################

