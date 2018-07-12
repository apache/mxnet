#!/usr/bin/env python
"""
modifed from DeformableConvNets (https://github.com/msracver/Deformable-ConvNets)
"""

"""
extensions to be built:
- gpu_nms
"""

import numpy as np
import os
from distutils.spawn import spawn, find_executable
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys

# make sure cl.exe can be found in PATH environment variable.
# If not, appending the binary directory of Visual Studio that contains cl.exe
# You may specify a different folder 
# If using different version of Visual Studio
PATH = os.environ.get('PATH')
if find_executable("cl.exe", PATH) is None:
    raise EnvironmentError('The compiler `cl.exe` could not be located in your $PATH. '
                'Please add directorys like "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin" to your $PATH.')

# CUDA specific config
# nvcc is assumed to be in user's PATH
nvcc_compile_args = ['-O', '--ptxas-options=-v', '-arch=compute_35', '-code=sm_35,sm_52,sm_61', '-c', '--compiler-options=-fPIC']
nvcc_compile_args = os.environ.get('NVCCFLAGS', '').split() + nvcc_compile_args
cuda_libs = ['cublas']


import distutils.msvc9compiler
distutils.msvc9compiler.VERSION = 14.0

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


cudamat_ext = Extension('gpu_nms',
                        sources=[
                                'gpu_nms.cu'
                                ],
                        language='c++',
                        libraries=cuda_libs,
                        extra_compile_args=nvcc_compile_args,
                        include_dirs = [numpy_include, 'C:\\Programming\\CUDA\\v8.0\\include'])


class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        if hasattr(self.compiler, '_c_extensions'):
            self.compiler._c_extensions.append('.cu')  # needed for Windows
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        elif self.compiler.compiler_type == 'msvc':
            # There are several things we need to do to change the commands
            # issued by MSVCCompiler into one that works with nvcc. In the end,
            # it might have been easier to write our own CCompiler class for
            # nvcc, as we're only interested in creating a shared library to
            # load with ctypes, not in creating an importable Python extension.
            # - First, we replace the cl.exe or link.exe call with an nvcc
            #   call. In case we're running Anaconda, we search cl.exe in the
            #   original search path we captured further above -- Anaconda
            #   inserts a MSVC version into PATH that is too old for nvcc.
            cmd[:1] = ['nvcc', '--compiler-bindir',
                       os.path.dirname(find_executable("cl.exe", PATH))
                       or cmd[0]]
            # - Secondly, we fix a bunch of command line arguments.
            for idx, c in enumerate(cmd):
                # create .dll instead of .pyd files
                #if '.pyd' in c: cmd[idx] = c = c.replace('.pyd', '.dll')  #20160601, by MrX
                # replace /c by -c
                if c == '/c': cmd[idx] = '-c'
                # replace /DLL by --shared
                elif c == '/DLL': cmd[idx] = '--shared'
                # remove --compiler-options=-fPIC
                elif '-fPIC' in c: del cmd[idx]
                # replace /Tc... by ...
                elif c.startswith('/Tc'): cmd[idx] = c[3:]
                # replace /Fo... by -o ...
                elif c.startswith('/Fo'): cmd[idx:idx+1] = ['-o', c[3:]]
                # replace /LIBPATH:... by -L...
                elif c.startswith('/LIBPATH:'): cmd[idx] = '-L' + c[9:]
                # replace /OUT:... by -o ...
                elif c.startswith('/OUT:'): cmd[idx:idx+1] = ['-o', c[5:]]
                # remove /EXPORT:initlibcudamat or /EXPORT:initlibcudalearn
                elif c.startswith('/EXPORT:'): del cmd[idx]
                # replace cublas.lib by -lcublas
                elif c == 'cublas.lib': cmd[idx] = '-lcublas'
            # - Finally, we pass on all arguments starting with a '/' to the
            #   compiler or linker, and have nvcc handle all other arguments
            if '--shared' in cmd:
                pass_on = '--linker-options='
                # we only need MSVCRT for a .dll, remove CMT if it sneaks in:
                cmd.append('/NODEFAULTLIB:libcmt.lib')
            else:
                pass_on = '--compiler-options='
            cmd = ([c for c in cmd if c[0] != '/'] +
                   [pass_on + ','.join(c for c in cmd if c[0] == '/')])
            # For the future: Apart from the wrongly set PATH by Anaconda, it
            # would suffice to run the following for compilation on Windows:
            # nvcc -c -O -o <file>.obj <file>.cu
            # And the following for linking:
            # nvcc --shared -o <file>.dll <file1>.obj <file2>.obj -lcublas
            # This could be done by a NVCCCompiler class for all platforms.
        spawn(cmd, search_path, verbose, dry_run)

setup(name="py_fast_rcnn_gpu",
      description="Performs linear algebra computation on the GPU via CUDA",
      ext_modules=[cudamat_ext],
      cmdclass={'build_ext': CUDA_build_ext},
)
