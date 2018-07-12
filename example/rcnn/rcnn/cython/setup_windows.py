# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn) and Deformable-ConvNets (https://github.com/msracver/Deformable-ConvNets)
# --------------------------------------------------------

"""
extensions to be built:
- bbox
- cpu_nms
"""

import os
import numpy as np
from setuptools import setup
from distutils.extension import Extension
from easydict import EasyDict as edict
from Cython.Distutils import build_ext
from pprint import pprint


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        "bbox",
        ["bbox.pyx"],
        extra_compile_args={'gcc': []},
        include_dirs=[numpy_include]
    ),
    Extension(
        "cpu_nms",
        ["cpu_nms.pyx"],
        extra_compile_args={'gcc': []},
        include_dirs = [numpy_include]
    )
]

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    #self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    #default_compiler_so = self.spawn 
    #default_compiler_so = self.rc
    super = self.compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
        postfix=os.path.splitext(sources[0])[1]
        
        if postfix == '.cu':
            # use the cuda for .cu files
            #self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        return super(sources, output_dir, macros, include_dirs, debug, extra_preargs, postargs, depends)
        # reset the default compiler_so, which we might have changed for cuda
        #self.rc = default_compiler_so

    # inject our redefined _compile method into the class
    self.compile = compile

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(
    name='frcnn_cython',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)
