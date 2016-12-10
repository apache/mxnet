# pylint: disable=invalid-name, exec-used
"""Setup mxnet package."""
from __future__ import absolute_import
import os
import sys
import subprocess
from setuptools import setup
from distutils.command.clean import clean as _clean
from distutils.command.build_py import build_py as _build_py
from distutils.spawn import find_executable

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']


PROTO_FILES = [
    'mxnet/tensorboard/types.proto',
    'mxnet/tensorboard/tensor_shape.proto',
    'mxnet/tensorboard/tensor.proto',
    'mxnet/tensorboard/resource_handle.proto',
    'mxnet/tensorboard/summary.proto',
    'mxnet/tensorboard/event.proto',
    ]


# Find the Protocol Compiler.
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = find_executable("protoc")


def generate_proto(source):
  """Invokes the Protocol Compiler to generate a _pb2.py from the given
  .proto file.  Does nothing if the output already exists and is newer than
  the input."""

  output = source.replace(".proto", "_pb2.py")

  if (not os.path.exists(output) or
      (os.path.exists(source) and
       os.path.getmtime(source) > os.path.getmtime(output))):
    print "Generating %s..." % output

    if not os.path.exists(source):
      sys.stderr.write("Can't find required file: %s\n" % source)
      sys.exit(-1)

    if protoc == None:
      sys.stderr.write(
          "Protocol buffers compiler 'protoc' not installed or not found.\n"
          )
      sys.exit(-1)

    protoc_command = [ protoc, "-I.", "--proto_path=mxnet/tensorboard", "--python_out=.", source ]  # TODO. prettify this
    if subprocess.call(protoc_command) != 0:
      sys.exit(-1)


class build_py(_build_py):
  def run(self):
    for f in PROTO_FILES:
        generate_proto(f)
    _build_py.run(self)


class clean(_clean):
  def run(self):
    # Delete generated files in the code tree.
    for (dirpath, dirnames, filenames) in os.walk("."):
      for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        if filepath.endswith("_pb2.py"):
          os.remove(filepath)
    # _clean is an old-style class, so super() doesn't work.
    _clean.run(self)


setup(name='mxnet',
      version=__version__,
      description=open(os.path.join(CURRENT_DIR, 'README.md')).read(),
      install_requires=[
          'numpy',
      ],
      zip_safe=False,
      packages=['mxnet', 'mxnet.module', 'mxnet.tensorboard'],
      cmdclass = {'clean': clean, 'build_py': build_py},
      data_files=[('mxnet', [LIB_PATH[0]])],
      url='https://github.com/dmlc/mxnet')
