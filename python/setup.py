"""Setup script for mxnet."""
from __future__ import absolute_import
import sys
from setuptools import setup
sys.path.insert(0, '.')
import mxnet

LIB_PATH = mxnet.base.find_lib_path()

setup(name='mxnet',
      version=mxnet.__version__,
      description=mxnet.__doc__,
      install_requires=[
          'numpy',
      ],
      zip_safe=False,
      packages=['mxnet'],
      data_files=[('mxnet', [LIB_PATH[0]])],
      url='https://github.com/dmlc/mxnet')
