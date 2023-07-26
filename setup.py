from setuptools import Extension, setup, find_packages
import subprocess
import os
from Cython.Build import cythonize
import numpy as np


def get_gsl_config(args):
    try:
        cmd = ["gsl-config"] + args
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return result.stdout.decode().strip().split(" ")[0][2:]
    except subprocess.CalledProcessError:
        return None


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


try:
    gsl_include_dir = get_gsl_config(["--cflags"])
    gsl_lib_dir = get_gsl_config(["--libs"])
except FileNotFoundError:
    print(
        "Could not find GSL, set the enviromental variables GSL_INCLUDE_DIR,GSL_LIB_DIR "
    )
    gsl_include_dir = os.environ.get("GSL_INCLUDE_DIR")
    gsl_lib_dir = os.environ.get("GSL_LIB_DIR")

if gsl_include_dir is None or gsl_lib_dir is None:
    raise FileNotFoundError("GSL not found!")

extensions = [
    Extension(
        "sbi_smfs.utils.gsl_spline",
        ["sbi_smfs/utils/gsl_spline.pyx"],
        include_dirs=[np.get_include(), gsl_include_dir],
        library_dirs=[gsl_lib_dir],
        libraries=["gsl", "gslcblas"],
        language_level=3,
    ),
    Extension(
        "sbi_smfs.simulator.brownian_integrator",
        ["sbi_smfs/simulator/brownian_integrator.pyx"],
        include_dirs=[np.get_include(), gsl_include_dir],
        library_dirs=[gsl_lib_dir],
        libraries=["gsl", "gslcblas"],
        language_level=3,
    ),
]


setup(
    name="SBIsmfs",
    version="0.0.1",
    author="Lars Dingeldein",
    author_email="dingeldein@fias.uni-frankfurt.de",
    description="Simulation-based inference for single-molecule force-spectroscopy",
    url="https://github.com/Dingel321/SBIsmfs.git",
    ext_modules=cythonize(extensions),
    packages=find_packages(),
    install_requires=get_requirements(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "generate_simulations=sbi_smfs.inference.command_line_tools:cmd_generate_simulations",
            "train_armortized_posterior=sbi_smfs.inference.command_line_tools:cmd_train_armortized_posterior",
            "train_sequential_posterior=sbi_smfs.inference.command_line_tools:cmd_train_sequential_posterior",
            "train_truncated_posterior=sbi_smfs.inference.command_line_tools:cmd_train_truncated_posterior",
        ],
    },
)
