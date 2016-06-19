# pylint: disable=invalid-name, exec-used
"""Setup mxnet package."""
from __future__ import absolute_import
import os
from setuptools import setup

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']

setup(name='mxnet',
      version=__version__,
      description=open(os.path.join(CURRENT_DIR, 'README.md')).read(),
      install_requires=[
          'numpy',
      ],
      zip_safe=False,
      packages=['mxnet', 'mxnet.module'],
      data_files=[('mxnet', [LIB_PATH[0]])],
      url='https://github.com/dmlc/mxnet')
