from gls_random cimport *
cimport numpy as cnp
import numpy as np


def gaussian_rn(int N):

    cdef gsl_rng_type * T
    cdef gsl_rng * r
    cdef int i
    cdef int s = 1
    cdef long seed = np.random.randint(2**63) # Initalize seed with numpy 2**63 = max(int64)
    cdef cnp.ndarray[double] numbers = np.empty(N, dtype=np.double)

    gsl_rng_env_setup()

    T = gsl_rng_default
    r = gsl_rng_alloc(T)
    gsl_rng_set(r, seed)

    cdef double k
    for i  from 0 <= i < N:
        k = gsl_ran_gaussian_ziggurat(r, s)
        numbers[i] = k

    gsl_rng_free (r)
    return numbers