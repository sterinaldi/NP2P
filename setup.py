import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

#ext_modules=[
#             Extension('parest.loglikelihood',
#                       sources=[os.path.join("parest", "loglikelihood.pyx")],
#                       libraries=["m"], # Unix-like specific
#                       extra_compile_args=["-O3","-ffast-math"],
#                       include_dirs=['parest', numpy.get_include()]
#                       ),
#             Extension('parest.models',
#                       sources=[os.path.join("parest", "models.pyx")],
#                       libraries=["m"], # Unix-like specific
#                       extra_compile_args=["-O3","-ffast-math"],
#                       include_dirs=['parest', numpy.get_include()]
#                       ),
#             ]
#             
#ext_modules = cythonize(ext_modules)

setup(
    name = 'parest',
    description = 'Parameter estimation from non-parametric inference',
    author = 'Stefano Rinaldi, Walter Del Pozzo',
    author_email = 'stefano.rinaldi@uni-heidelberg.de, walter.delpozzo@unipi.it',
    url = 'https://github.com/sterinaldi/parametric',
    python_requires = '>=3.7',
    packages = ['parest'],
    include_dirs = [numpy.get_include()],
    install_requires=requirements,
    setup_requires=['numpy'],
    entry_points={},
#    package_data={"": ['*.c', '*.pyx', '*.pxd']},
#    ext_modules=ext_modules,
    )
