# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
    kwargs = {'install_requires': ['numpy<=1.13.3,>=1.8.2', 'requests==2.18.4', 'graphviz==0.8.1'], 'zip_safe': False}
from setuptools import find_packages

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

sys.path.insert(0, CURRENT_DIR)

# Try to generate auto-complete code
try:
    from mxnet.base import _generate_op_module_signature
    from mxnet.ndarray.register import _generate_ndarray_function_code
    from mxnet.symbol.register import _generate_symbol_function_code
    _generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
    _generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)
except: # pylint: disable=bare-except
    pass

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
      packages=find_packages(),
      data_files=[('mxnet', [LIB_PATH[0]])],
      url='https://github.com/apache/incubator-mxnet',
      ext_modules=config_cython(),
      **kwargs)
