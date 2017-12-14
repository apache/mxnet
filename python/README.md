MXNet Python Package
====================
MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to mix the flavours of deep learning programs together to maximize the efficiency and your productivity.

This directory and nested files contain MXNet Python package and language binding.

## Installation
To install MXNet Python package, visit MXNet [Install Instruction](http://mxnet.incubator.apache.org/get_started/install.html)


## Running the unit tests

For running unit tests, you will need the [nose PyPi package](https://pypi.python.org/pypi/nose). To install:
```bash
pip install nose
```

Once ```nose``` is installed, run the following from MXNet root directory:

```
nosetests tests/python/unittest
nosetests tests/python/train

```