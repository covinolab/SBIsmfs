from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'brownian_integrator',
        ['scr/simulator/brownian_integrator.pyx'],
        include_dirs=[np.get_include()],
        libraries=['gsl']
    ),
    Extension(
        'gls_random_gen',
        ['scr/utils/gls_random_gen.pyx'],
        include_dirs=[np.get_include()],
        libraries=['gsl']
    ),
    Extension(
        'gls_spline',
        ['scr/utils/gls_spline.pyx'],
        include_dirs=[np.get_include()],
        libraries=['gsl']
    )
]

setup(
    ext_modules=cythonize(extensions),
)
