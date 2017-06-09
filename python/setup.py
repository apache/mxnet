# pylint: disable=invalid-name, exec-used
"""Setup mxnet package."""
from __future__ import absolute_import
import os
import sys
# need to use distutils.core for correct placement of cython dll
kwargs = {}
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension
    kwargs = {'install_requires': ['numpy', 'requests', 'graphviz'], 'zip_safe': False}

with_cython = False
if '--with-cython' in sys.argv:
    with_cython = True
    sys.argv.remove('--with-cython')

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']


def config_cython():
    """Try to configure cython and return cython configuration"""
    if not with_cython:
        return []
    # pylint: disable=unreachable
    if os.name == 'nt':
        print("WARNING: Cython is not supported on Windows, will compile without cython module")
        return []

    try:
        from Cython.Build import cythonize
        # from setuptools.extension import Extension
        if sys.version_info >= (3, 0):
            subdir = "_cy3"
        else:
            subdir = "_cy2"
        ret = []
        path = "mxnet/cython"
        if os.name == 'nt':
            library_dirs = ['mxnet', '../build/Release', '../build']
            libraries = ['libmxnet']
        else:
            library_dirs = None
            libraries = None

        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "mxnet/%s/.%s" % (subdir, fn[:-4]),
                ["mxnet/cython/%s" % fn],
                include_dirs=["../include/", "../nnvm/include"],
                library_dirs=library_dirs,
                libraries=libraries,
                language="c++"))
        return cythonize(ret)
    except ImportError:
        print("WARNING: Cython is not installed, will compile without cython module")
        return []


setup(name='mxnet',
      version=__version__,
      description=open(os.path.join(CURRENT_DIR, 'README.md')).read(),
      packages=[
          'mxnet', 'mxnet.module', 'mxnet._ctypes', 'mxnet.rnn',
          'mxnet._cy2', 'mxnet._cy3', 'mxnet.notebook', 'mxnet.contrib'
          ],
      data_files=[('mxnet', [LIB_PATH[0]])],
      url='https://github.com/dmlc/mxnet',
      ext_modules=config_cython(),
      **kwargs)
