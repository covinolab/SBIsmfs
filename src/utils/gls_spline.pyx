from gls_spline cimport *
from math import *
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp


def gls_spline(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        cnp.ndarray[double] x_eval
        ):

    cdef int N_knots = len(x_knots)
    cdef int N_eval = len(x_eval)

    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i  from 0 <= i < N_knots:
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    cdef gsl_interp_accel *acc
    acc = gsl_interp_accel_alloc()
    cdef gsl_spline *spline
    spline = gsl_spline_alloc(gsl_interp_cspline, N_knots)

    gsl_spline_init(spline, x_k, y_k, N_knots)

    cdef cnp.ndarray[double] y_eval = np.empty(N_eval, dtype=np.double)
    cdef tmp_y

    for i  from 0 <= i < N_eval:
        tmp_y = gsl_spline_eval(spline, x_eval[i], acc)
        y_eval[i] = tmp_y

    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    return y_eval


def gls_spline_der(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        cnp.ndarray[double] x_eval
        ):

    cdef int N_knots = len(x_knots)
    cdef int N_eval = len(x_eval)

    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i  from 0 <= i < N_knots:
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    cdef gsl_interp_accel *acc
    acc = gsl_interp_accel_alloc ()
    cdef gsl_spline *spline
    spline = gsl_spline_alloc(gsl_interp_cspline, N_knots)

    gsl_spline_init(spline, x_k, y_k, N_knots)

    cdef cnp.ndarray[double] y_eval = np.empty(N_eval, dtype=np.double)
    cdef tmp_y

    for i  from 0 <= i < N_eval:
        tmp_y = gsl_spline_eval_deriv(spline, x_eval[i], acc)
        y_eval[i] = tmp_y

    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    return y_eval