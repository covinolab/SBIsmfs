import cython
import numpy as np

from libc.math cimport sqrt
cimport numpy as cnp
from sbi_smfs.utils.gsl_spline cimport *
from gsl_random_numbers cimport *
from libc.stdlib cimport malloc, free
from libc.time cimport time


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def brownian_integrator(
        double x0,
        double q0,
        double Dx,
        double Dq,
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double k,
        long N,
        double dt,
        int fs
        ):
    """Integrator for constant force SMFE.
    Molecular free energy profile is contructed by a cubic spline.
    
    Parameters
    ----------
    double x0 : float
        Initial x position.
    double q0 : float
        Intial q position.
    double Dx : float
        Diffusion constant along x.
    double Dq : float
        Diffusion constant along q.
    cnp.ndarray[double] x_knots : np.Array
        x - positions of spline knots.
    cnp.ndarray[double] y_knots : np.Array
        y - position of spline knots.
    double k : float
        Spring konstant of linker.
    long N : int
        Number of iterations.
    double dt : float
        Time step.
    int fs : int
        Saving frequency of postions.
    
    Returns
    -------
    q : np.Array
        q-Trajectory.
    
    """
    
    # Initialize ints
    cdef int i
    cdef int N_knots = len(x_knots)
    cdef long N_save = N // fs

    # Initalize random number generatot and seed
    cdef gsl_rng_type * T
    cdef gsl_rng * r
    cdef long seed = np.random.randint(low=1, high=2**63) # Initalize seed with randints between 1 and 2**63 = max(int64)

    gsl_rng_env_setup()
    T = gsl_rng_default
    r = gsl_rng_alloc(T)
    gsl_rng_set(r, seed)

    # initalize spline knots as c type arrays
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    # Transfer spline knots from numpy to c arry
    for i from 0 <= i < N_knots:
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    # Calculation of spline interpolation
    cdef gsl_interp_accel *acc
    acc = gsl_interp_accel_alloc ()
    cdef gsl_spline *spline
    spline = gsl_spline_alloc(gsl_interp_cspline, N_knots)
    gsl_spline_init(spline, x_k, y_k, N_knots)

    # Initialize constant for integrator
    cdef double Ax = Dx * dt
    cdef double Bx = sqrt(2 * Ax)
    cdef double Aq = Dq * dt
    cdef double Bq = sqrt(2 * Aq)
    cdef double xold = x0
    cdef double qold = q0
    cdef double Fx, Fq, xnew, qnew

    # Initialize ndarray to save trajectory
    cdef cnp.ndarray[double] q = np.empty(N_save, dtype=np.double)
    q[0] = q0

    # Main integration loop
    for i in range(1, N):

        # Forces evaluation
        Fx = -gsl_spline_eval_deriv(spline, xold, acc) - k * (xold - qold)
        Fq = k * (xold - qold)

        # integration + random number gen
        xnew = xold + Ax * Fx + Bx * gsl_ran_gaussian_ziggurat(r, 1)
        qnew = qold + Aq * Fq + Bq * gsl_ran_gaussian_ziggurat(r, 1)

        # Save position
        if (i % fs) == 0:
            q[i / fs] = qnew

        xold = xnew
        qold = qnew

    # Free allocated memory
    gsl_spline_free (spline)
    gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    return q

