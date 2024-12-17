import cython
import numpy as np

from libc.math cimport sqrt
cimport numpy as cnp
cimport sbi_smfs.utils.gsl_spline as c_spline
cimport sbi_smfs.simulator.gsl_random_numbers as grn
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
        double kt,
        long N,
        double dt,
        int fs
        ):
    """
    Integrator for constant force SMFE.
    Molecular free energy profile is constructed by a cubic spline.
    
    Parameters
    ----------
    double x0 : float
        Initial x position.
    double q0 : float
        Initial q position.
    double Dx : float
        Diffusion constant along x.
    double Dq : float
        Diffusion constant along q.
    cnp.ndarray[double] x_knots : np.Array
        x - positions of spline knots.
    cnp.ndarray[double] y_knots : np.Array
        y - position of spline knots.
    double k : float
        Spring constant of linker.
    long N : int
        Number of iterations.
    double dt : float
        Time step.
    int fs : int
        Saving frequency of positions.
    
    Returns
    -------
    q : np.Array
        q-Trajectory.
    
    """
    
    # Initialize ints
    cdef int i
    cdef int N_knots = len(x_knots)
    cdef long N_save = N // fs

    # Initalize random number generator and seed
    cdef grn.gsl_rng_type * T
    cdef grn.gsl_rng * r
    cdef long seed = np.random.randint(low=1, high=2**63) # Initialize seed with random integers between 1 and 2**63

    grn.gsl_rng_env_setup()
    T = grn.gsl_rng_default
    r = grn.gsl_rng_alloc(T)
    grn.gsl_rng_set(r, seed)

    # initalize spline knots as c type arrays
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    # Transfer spline knots from numpy to C array
    for i in range(N_knots):
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    # Calculation of spline interpolation
    cdef c_spline.gsl_interp_accel *acc
    acc = c_spline.gsl_interp_accel_alloc ()
    cdef c_spline.gsl_spline *spline
    spline = c_spline.gsl_spline_alloc(c_spline.gsl_interp_cspline, N_knots)
    c_spline.gsl_spline_init(spline, x_k, y_k, N_knots)

    # Initialize constant for integrator
    cdef double Ax = Dx * dt
    cdef double Bx = sqrt(2.0 * Ax)
    cdef double Aq = Dq * dt
    cdef double Bq = sqrt(2.0 * Aq)
    cdef double xold = x0
    cdef double qold = q0
    cdef double Fx, Fq, xnew, qnew, spline_deriv
    cdef int status

    # Initialize ndarray to save trajectory
    cdef cnp.ndarray[double] q = np.empty(N_save, dtype=np.double)
    q[0] = q0

    # Main integration loop
    for i in range(1, N):

        # Forces evaluation
        status = c_spline.gsl_spline_eval_deriv_e(spline, xold, acc, &spline_deriv)
        if status != 0:
            break

        Fx = -spline_deriv - k * (xold - qold)
        Fq = k * (xold - qold) - kt * qold

        # integration + random number gen
        xnew = xold + Ax * Fx + Bx * grn.gsl_ran_gaussian_ziggurat(r, 1.0)
        qnew = qold + Aq * Fq + Bq * grn.gsl_ran_gaussian_ziggurat(r, 1.0)

        # Save position
        if (i % fs) == 0:
            q[i / fs] = xnew

        xold = xnew
        qold = qnew

    # Free allocated memory
    c_spline.gsl_spline_free (spline)
    c_spline.gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    # If the status is not 0, it indicates an error in the spline evaluation.
    # Returning None to indicate that the integration process failed.
    if status != 0:
        return None
    return q

