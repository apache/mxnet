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

# coding: utf-8
# pylint: disable=invalid-name, exec-used
"""Setup mxnet package for pip."""
from __future__ import absolute_import
from datetime import datetime
import os
import sys
import shutil
import platform

if platform.system() == 'Linux':
    sys.argv.append('--universal')
    sys.argv.append('--plat-name=manylinux1_x86_64')

from setuptools import setup, find_packages
from setuptools.dist import Distribution

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'mxnet-build/python/mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']
if 'TRAVIS_TAG' not in os.environ or not os.environ['TRAVIS_TAG'].strip():
    __version__ += 'b{0}'.format(datetime.today().strftime('%Y%m%d'))
elif 'TRAVIS_TAG' in os.environ and os.environ['TRAVIS_TAG'].startswith('patch-'):
    __version__ = os.environ['TRAVIS_TAG'].split('-')[1]

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return platform.system() == 'Darwin'


DEPENDENCIES = [
    'numpy>1.16.0,<2.0.0',
    'requests>=2.20.0',
    'graphviz<0.9.0,>=0.8.1'
]

shutil.rmtree(os.path.join(CURRENT_DIR, 'mxnet'), ignore_errors=True)
shutil.rmtree(os.path.join(CURRENT_DIR, 'dmlc_tracker'), ignore_errors=True)
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/python/mxnet'),
                os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/dmlc-core/tracker/dmlc_tracker'),
                os.path.join(CURRENT_DIR, 'dmlc_tracker'))
shutil.copy(LIB_PATH[0], os.path.join(CURRENT_DIR, 'mxnet'))

# copy license and notice
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/licenses'),
            os.path.join(CURRENT_DIR, 'mxnet/licenses'))

# copy tools to mxnet package
shutil.rmtree(os.path.join(CURRENT_DIR, 'mxnet/tools'), ignore_errors=True)
os.mkdir(os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/launch.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/im2rec.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/kill-mxnet.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/parse_log.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/diagnose.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/tools/caffe_converter'), os.path.join(CURRENT_DIR, 'mxnet/tools/caffe_converter'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/tools/bandwidth'), os.path.join(CURRENT_DIR, 'mxnet/tools/bandwidth'))

# copy headers to mxnet package
shutil.rmtree(os.path.join(CURRENT_DIR, 'mxnet/include'), ignore_errors=True)
os.mkdir(os.path.join(CURRENT_DIR, 'mxnet/include'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/include/mxnet'),
                os.path.join(CURRENT_DIR, 'mxnet/include/mxnet'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/dlpack/include/dlpack'),
                os.path.join(CURRENT_DIR, 'mxnet/include/dlpack'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/dmlc-core/include/dmlc'),
                os.path.join(CURRENT_DIR, 'mxnet/include/dmlc'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/mshadow/mshadow'),
                os.path.join(CURRENT_DIR, 'mxnet/include/mshadow'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/tvm/nnvm/include/nnvm'),
                os.path.join(CURRENT_DIR, 'mxnet/include/nnvm'))

package_name = 'mxnet'

variant = os.environ['mxnet_variant'].upper()
if variant != 'CPU':
    package_name = 'mxnet_{0}'.format(variant.lower())

with open('doc/PYPI_README.md') as readme_file:
    long_description = readme_file.read()

with open('doc/{0}_ADDITIONAL.md'.format(variant)) as variant_doc:
    long_description = long_description + variant_doc.read()

# pypi only supports rst, so use pandoc to convert
import pypandoc
if platform.system() == 'Darwin':
    pypandoc.download_pandoc()
long_description = pypandoc.convert_text(long_description, 'rst', 'md')
short_description = 'MXNet is an ultra-scalable deep learning framework.'
libraries = []
if variant == 'CPU':
    libraries.append('openblas')
else:
    if variant.startswith('CU100'):
        libraries.append('CUDA-10.0')
    elif variant.startswith('CU92'):
        libraries.append('CUDA-9.2')
    elif variant.startswith('CU91'):
        libraries.append('CUDA-9.1')
    elif variant.startswith('CU90'):
        libraries.append('CUDA-9.0')
    elif variant.startswith('CU80'):
        libraries.append('CUDA-8.0')
    elif variant.startswith('CU75'):
        libraries.append('CUDA-7.5')
    if variant.endswith('MKL'):
        libraries.append('MKLDNN')

short_description += ' This version uses {0}.'.format(' and '.join(libraries))

package_data = {'mxnet': [os.path.join('mxnet', os.path.basename(LIB_PATH[0]))],
                'dmlc_tracker': []}
if variant.endswith('MKL'):
    if platform.system() == 'Darwin':
        shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libmklml.dylib'), os.path.join(CURRENT_DIR, 'mxnet'))
        shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libiomp5.dylib'), os.path.join(CURRENT_DIR, 'mxnet'))
        shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libmkldnn.0.dylib'), os.path.join(CURRENT_DIR, 'mxnet'))
        package_data['mxnet'].append('mxnet/libmklml.dylib')
        package_data['mxnet'].append('mxnet/libiomp5.dylib')
        package_data['mxnet'].append('mxnet/libmkldnn.0.dylib')
    else:
        shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libmklml_intel.so'), os.path.join(CURRENT_DIR, 'mxnet'))
        shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libiomp5.so'), os.path.join(CURRENT_DIR, 'mxnet'))
        shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libmkldnn.so.0'), os.path.join(CURRENT_DIR, 'mxnet'))
        package_data['mxnet'].append('mxnet/libmklml_intel.so')
        package_data['mxnet'].append('mxnet/libiomp5.so')
        package_data['mxnet'].append('mxnet/libmkldnn.so.0')
    shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/mkldnn/build/install/include'),
                    os.path.join(CURRENT_DIR, 'mxnet/include/mkldnn'))
if platform.system() == 'Linux':
    shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libgfortran.so.3'), os.path.join(CURRENT_DIR, 'mxnet'))
    package_data['mxnet'].append('mxnet/libgfortran.so.3')
    shutil.copy(os.path.join(os.path.dirname(LIB_PATH[0]), 'libquadmath.so.0'), os.path.join(CURRENT_DIR, 'mxnet'))
    package_data['mxnet'].append('mxnet/libquadmath.so.0')

# Copy licenses and notice
for f in os.listdir('mxnet/licenses'):
  package_data['mxnet'].append('mxnet/licenses/{}'.format(f))

from mxnet.base import _generate_op_module_signature
from mxnet.ndarray.register import _generate_ndarray_function_code
from mxnet.symbol.register import _generate_symbol_function_code
_generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
_generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)

setup(name=package_name,
      version=__version__,
      long_description=long_description,
      description=short_description,
      zip_safe=False,
      packages=find_packages(),
      package_data=package_data,
      include_package_data=True,
      install_requires=DEPENDENCIES,
      distclass=BinaryDistribution,
      license='Apache 2.0',
      classifiers=[ # https://pypi.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: C++',
          'Programming Language :: Cython',
          'Programming Language :: Other',  # R, Scala
          'Programming Language :: Perl',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      url='https://github.com/apache/incubator-mxnet')
