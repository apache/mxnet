from distutils.core import setup, Extension
import numpy
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "log_uniform",                                # the extension name
           sources=["log_uniform.pyx", "Log_Uniform_Sampler.cpp"], # the Cython source and additional C++ source files
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-std=c++11"],
           include_dirs=[numpy.get_include()]
      )))
