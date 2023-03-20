from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

path_to_gsl_include = '/opt/homebrew/Cellar/gsl/2.7.1/include'
path_to_gsl_lib = '/opt/homebrew/Cellar/gsl/2.7.1/lib'

extensions = [
    Extension(
        'src.utils.gls_random_gen',
        ['src/utils/gls_random_gen.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas']
         ),
    Extension(
        'src.utils.gls_spline',
        ['src/utils/gls_spline.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas']
    ),
    Extension(
        'src.simulator.brownian_integrator',
        ['src/simulator/brownian_integrator.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas']
    )
]

setup(
    ext_modules=cythonize(extensions)
)
