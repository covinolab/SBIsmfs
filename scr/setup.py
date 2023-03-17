from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'brownian_integrator',
        ['brownian_integrator.pyx'],
        include_dirs=[np.get_include()],
        libraries=['gsl']
    ),
    Extension(
        'gls_random_gen',
        ['gls_random_gen.pyx'],
        include_dirs=[np.get_include()],
        libraries=['gsl']
    ),
    Extension(
        'gls_spline',
        ['gls_spline.pyx'],
        include_dirs=[np.get_include()],
        libraries=['gsl']
    )
]

setup(
    ext_modules=cythonize(extensions),
)
