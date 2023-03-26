from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy as np

path_to_gsl_include = '/opt/homebrew/Cellar/gsl/2.7.1/include'
path_to_gsl_lib = '/opt/homebrew/Cellar/gsl/2.7.1/lib'

extensions = [
    Extension(
        'sbi_smfs.utils.gls_spline',
        ['sbi_smfs/utils/gls_spline.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas'],
        language_level=3
    ),
    Extension(
        'sbi_smfs.simulator.brownian_integrator',
        ['sbi_smfs/simulator/brownian_integrator.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas'],
        language_level=3
    )
]

requirements = [
    "bottleneck",
    "torch",
    "sbi",
    "matplotlib",
    "cython",
    "numpy",
    "numba",
    "setuptools"
]

setup(
    name='SBIsmfs',
    version='0.0.1',
    author='Lars Dingeldein',
    author_email='dingeldein@fias.uni-frankfurt.de',
    description='Simulation-based inference for single-molecule force-spectroscopy',
    url='https://github.com/Dingel321/SBIsmfs.git',
    ext_modules=cythonize(extensions),
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False
)
