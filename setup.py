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
        libraries=['gsl', 'gslcblas']
    ),
    Extension(
        'sbi_smfs.simulator.brownian_integrator',
        ['sbi_smfs/simulator/brownian_integrator.pyx'],
        include_dirs=[np.get_include(), path_to_gsl_include],
        library_dirs=[path_to_gsl_lib],
        libraries=['gsl', 'gslcblas']
    )
]

setup(
    name='SBIsmfs',
    version='0.0.1',
    description='Simulation-based inference for single-molecule force-spectroscopy',
    url='https://github.com/Dingel321/SBIsmfs.git',
    ext_modules=cythonize(extensions),
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'scipy', 'numba'],
    zip_safe=False
)
