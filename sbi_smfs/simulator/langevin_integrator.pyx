import cython
import numpy as np

from libc.math cimport sqrt
cimport numpy as cnp
cimport sbi_smfs.utils.gsl_spline as c_spline
cimport sbi_smfs.simulator.gsl_random_numbers as grn
from libc.stdlib cimport malloc, free
from libc.time cimport time

# code adapted from https://github.com/synapticarbors/pylangevin-integrator
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def langevin_integrator(
        double x0,
        double q0,
        double xm,
        double xnu,
        double Dq,
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double k,
        long N,
        double dt,
        int fs
        ):
    
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
    cdef double Aq = Dq * dt
    cdef double Bq = sqrt(2.0 * Aq)
    cdef double xold = x0
    cdef double qold = q0
    cdef double vxold = 0.0
    cdef double Fx, Fq, xnew, qnew, spline_deriv
    cdef double beta = 1.0
    cdef double sigma = sqrt(2.0 * xnu / (beta * xm))
    cdef double b1 = 1.0 - 0.5 * dt * xnu + 0.125 * (dt ** 2) * (xnu ** 2)
    cdef double b2 = 0.5 * dt - 0.125 * xnu * dt ** 2
    cdef double s3 = sqrt(3.0)
    cdef double sdt3 = sigma*sqrt(dt ** 3)
    cdef double sdt = sigma*sqrt(dt)
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
        Fx = Fx / xm
        Fq = k * (xold - qold)

        xi = grn.gsl_ran_gaussian_ziggurat(r, 1.0)
        eta = grn.gsl_ran_gaussian_ziggurat(r, 1.0)
        n1 = 0.5 * sdt * xi
        n3 = 0.5 * sdt3 * eta / s3
        n4 = sdt3 * (0.125 * xi + 0.25 * eta / s3)
        n5 = n1 - xnu * n4

        vxold = vxold * b1 + Fx * b2 + n5
        vxnew = vxold * b1 + Fx * b2 + n5 
        xnew = xold + dt * vxold + n3
        qnew = qold + Aq * Fq + Bq * grn.gsl_ran_gaussian_ziggurat(r, 1.0)
    
        # Save position
        if (i % fs) == 0:
            q[i / fs] = qnew

        xold = xnew
        vxold = vxnew
        qold = qnew

    # Free allocated memory
    c_spline.gsl_spline_free (spline)
    c_spline.gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    if status != 0:
        return None
    return q