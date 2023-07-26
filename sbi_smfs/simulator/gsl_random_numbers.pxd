cdef extern from "gsl/gsl_rng.h":

    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    cdef gsl_rng_type *gsl_rng_default
    unsigned long int gsl_rng_default_seed
    gsl_rng *gsl_rng_alloc( gsl_rng_type * T) nogil

    void gsl_rng_free(gsl_rng * r) nogil
    void gsl_rng_set( gsl_rng * r, unsigned long int seed) nogil

    gsl_rng_type * gsl_rng_env_setup () nogil



cdef extern from "gsl/gsl_randist.h":

    double gsl_ran_gaussian( gsl_rng * r,  double sigma) nogil
    double gsl_ran_gaussian_ziggurat( gsl_rng *r, double sigma) nogil
