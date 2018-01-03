MXNet Python Package
====================
This directory and nested files contain MXNet Python package and language binding.

## Installation
To install MXNet Python package, visit MXNet [Install Instruction](http://mxnet.incubator.apache.org/install/index.html)


## Running the unit tests

For running unit tests, you will need the [nose PyPi package](https://pypi.python.org/pypi/nose). To install:
```bash
pip install --upgrade nose
```

Once ```nose``` is installed, run the following from MXNet root directory:
```
nosetests tests/python/unittest
nosetests tests/python/train

```