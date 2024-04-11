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
        long N,
        double dt,
        int fs
        ):
    """
    Integrator for constant force SMFE.
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
    cdef grn.gsl_rng_type * T
    cdef grn.gsl_rng * r
    cdef long seed = np.random.randint(low=1, high=2**63) # Initalize seed with randints between 1 and 2**63 = max(int64)

    grn.gsl_rng_env_setup()
    T = grn.gsl_rng_default
    r = grn.gsl_rng_alloc(T)
    grn.gsl_rng_set(r, seed)

    # initalize spline knots as c type arrays
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    # Transfer spline knots from numpy to c arry
    for i from 0 <= i < N_knots: # TODO : deprecated syntax
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
        Fq = k * (xold - qold)

        # integration + random number gen
        xnew = xold + Ax * Fx + Bx * grn.gsl_ran_gaussian_ziggurat(r, 1.0)
        qnew = qold + Aq * Fq + Bq * grn.gsl_ran_gaussian_ziggurat(r, 1.0)

        # Save position
        if (i % fs) == 0:
            q[i / fs] = qnew

        xold = xnew
        qold = qnew

    # Free allocated memory
    c_spline.gsl_spline_free (spline)
    c_spline.gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    if status != 0:
        return None
    return q


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def pdd_brownian_integrator(
        double x0,
        double q0,
        cnp.ndarray[double] Dx,
        double Dq,
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double k,
        long N,
        double dt,
        int fs
        ):
    """
    Integrator for constant force SMFE using position dependent Dx.
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
    cdef grn.gsl_rng_type * T
    cdef grn.gsl_rng * r
    cdef long seed = np.random.randint(low=1, high=2**63) # Initalize seed with randints between 1 and 2**63 = max(int64)

    grn.gsl_rng_env_setup()
    T = grn.gsl_rng_default
    r = grn.gsl_rng_alloc(T)
    grn.gsl_rng_set(r, seed)

    # initalize spline knots as c type arrays
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *Dx_k = <double *> malloc(N_knots * sizeof(double))

    # Transfer spline knots from numpy to c arry
    for i from 0 <= i < N_knots: # TODO : deprecated syntax
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]
        Dx_k[i] = Dx[i]

    # Calculation of spline interpolation
    cdef c_spline.gsl_interp_accel *acc_gx
    acc_gx = c_spline.gsl_interp_accel_alloc ()
    cdef c_spline.gsl_spline *spline_gx
    spline_gx = c_spline.gsl_spline_alloc(c_spline.gsl_interp_cspline, N_knots)
    c_spline.gsl_spline_init(spline_gx, x_k, y_k, N_knots)

    cdef c_spline.gsl_interp_accel *acc_dx
    acc_dx = c_spline.gsl_interp_accel_alloc ()
    cdef c_spline.gsl_spline *spline_dx
    spline_dx = c_spline.gsl_spline_alloc(c_spline.gsl_interp_cspline, N_knots)
    c_spline.gsl_spline_init(spline_dx, x_k, Dx_k, N_knots)


    # Initialize constant for integrator
    cdef double xold = x0
    cdef double qold = q0
    cdef double Ax
    cdef double Bx
    cdef double Aq = Dq * dt
    cdef double Bq = sqrt(2.0 * Aq)
    cdef double Fx, Fq, xnew, qnew, spline_gx_deriv, spline_dx_val, spline_dx_deriv
    cdef int status

    # Initialize ndarray to save trajectory
    cdef cnp.ndarray[double] q = np.empty(N_save, dtype=np.double)
    q[0] = q0

    # Main integration loop
    for i in range(1, N):

        # Forces evaluation
        status_gx_deriv = c_spline.gsl_spline_eval_deriv_e(spline_gx, xold, acc_gx, &spline_gx_deriv)
        status_dx = c_spline.gsl_spline_eval_e(spline_dx, xold, acc_dx, &spline_dx_val)
        status_dx_deriv = c_spline.gsl_spline_eval_deriv_e(spline_dx, xold, acc_dx, &spline_dx_deriv)
        if status_gx_deriv != 0 or status_dx != 0 or status_dx_deriv != 0:
            break

        Fx = -spline_gx_deriv - k * (xold - qold)
        Fq = k * (xold - qold)

        Ax = spline_dx_val * dt
        Bx = sqrt(2.0 * Ax)
        Cx = spline_dx_deriv * dt

        # integration + random number gen
        xnew = xold + Ax * Fx + Bx * grn.gsl_ran_gaussian_ziggurat(r, 1.0) + Cx
        qnew = qold + Aq * Fq + Bq * grn.gsl_ran_gaussian_ziggurat(r, 1.0)

        # Save position
        if (i % fs) == 0:
            q[i / fs] = qnew

        xold = xnew
        qold = qnew

    # Free allocated memory
    c_spline.gsl_spline_free (spline_dx)
    c_spline.gsl_spline_free (spline_gx)
    c_spline.gsl_interp_accel_free (acc_dx)
    c_spline.gsl_interp_accel_free (acc_gx)
    free(x_k)
    free(y_k)
    free(Dx_k)

    if status_dx_deriv != 0 or status_dx != 0 or status_dx_deriv != 0:
        return None
    return q

