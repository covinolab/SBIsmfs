import cython
cimport sbi_smfs.utils.gsl_spline as c_spline
from math import *
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def c_spline(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        cnp.ndarray[double] x_eval
        ):
    """Cython wrapper for GSL spline interpolation
    
    Parameters
    ----------
    x_knots : array_like
        x values of the knots
    y_knots : array_like
        y values of the knots
    x_eval : array_like
        x values where the spline is evaluated
    
    Returns
    -------
    y_eval : array_like
        y values of the spline at x_eval
    """

    cdef int N_knots = len(x_knots)
    cdef int N_eval = len(x_eval)

    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i  from 0 <= i < N_knots:
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    cdef c_spline.gsl_interp_accel *acc
    acc = c_spline.gsl_interp_accel_alloc()
    cdef c_spline.gsl_spline *spline
    spline = c_spline.gsl_spline_alloc(c_spline.gsl_interp_cspline, N_knots)

    c_spline.gsl_spline_init(spline, x_k, y_k, N_knots)

    cdef cnp.ndarray[double] y_eval = np.empty(N_eval, dtype=np.double)
    cdef double tmp_y
    cdef int status

    for i  from 0 <= i < N_eval:
        status = c_spline.gsl_spline_eval_e(spline, x_eval[i], acc, &tmp_y)
        if status != 0:
            break
        y_eval[i] = tmp_y

    c_spline.gsl_spline_free (spline)
    c_spline.gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    if status != 0:
        return None

    return y_eval


@cython.boundscheck(False)
@cython.wraparound(False)
def c_spline_der(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        cnp.ndarray[double] x_eval
        ):
    """Cython wrapper for GSL spline interpolation.
    
    Parameters
    ----------
    x_knots : array_like
        x values of the knots
    y_knots : array_like
        y values of the knots
    x_eval : array_like
        x values where the spline is evaluated
    
    Returns
    -------
    y_eval : array_like
        y values of the spline at x_eval
    """

    cdef int N_knots = len(x_knots)
    cdef int N_eval = len(x_eval)

    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i  from 0 <= i < N_knots:
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    cdef c_spline.gsl_interp_accel *acc
    acc = c_spline.gsl_interp_accel_alloc ()
    cdef c_spline.gsl_spline *spline
    spline = c_spline.gsl_spline_alloc(c_spline.gsl_interp_cspline, N_knots)

    c_spline.gsl_spline_init(spline, x_k, y_k, N_knots)

    cdef cnp.ndarray[double] y_eval = np.empty(N_eval, dtype=np.double)
    cdef double tmp_y
    cdef int status

    for i  from 0 <= i < N_eval:
        status = c_spline.gsl_spline_eval_deriv_e(spline, x_eval[i], acc, &tmp_y)
        if status != 0:
            break
        y_eval[i] = tmp_y

    c_spline.gsl_spline_free (spline)
    c_spline.gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    if status != 0:
        return None

    return y_eval