from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

path_to_gsl_include = 'path/to/gsl/include'
path_to_gsl_lib = 'path/to/gsl/lib'

extensions = [
    Extension(
        'brownian_integrator',
        ['scr/simulator/brownian_integrator.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas']
    ),
    Extension(
        'gls_random_gen',
        ['scr/utils/gls_random_gen.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas']
    ),
    Extension(
        'gls_spline',
        ['scr/utils/gls_spline.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas']
    )
]

setup(
    ext_modules=cythonize(extensions),
    compiler_directives={'language_level' : "3"}
)
