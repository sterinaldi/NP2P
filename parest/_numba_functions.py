import numpy as np
from numba import njit
from numba.extending import get_cython_function_address
import ctypes

"""
See https://stackoverflow.com/a/54855769
Wrapper (based on https://github.com/numba/numba/issues/3086) for scipy's cython implementation of gammaln.
"""

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@njit
def gammaln_jit(x):
    return gammaln_float64(x)

@njit
def gammaln_jit_vect(x):
    return np.array([gammaln_float64(xi) for xi in x])

@njit
def logsumexp_jit(a):
    a_max = np.max(a)
    return np.log(np.sum(np.exp(a - a_max))) + a_max
