Build and Installation
======================

Minimal system requirement:

- recent c++ compiler supporting C++ 11 such as `g++ >= 4.8`
- git
- BLAS library.
- opencv

On Ubuntu >= 13.10, one can install them by

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libblas-dev libopencv-dev
```

Then build mxnet

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; make -j4
```

To install the python package, first make sure `python >= 2.7` and `numpy >= ?` are installed, then

```bash
cd python; python setup.py install
```

If anything goes well, now we can train a multilayer perceptron on the hand
digit recognition dataset.

```bash
cd ..; python example/mnist/mlp.py
```

Advanced Build
--------------

- update the repo:

```bash
git pull
git submodule update
```

- install python package in developing model,

```bash
cd python; python setup.py develop --user
```

- modify the compiling options such as compilers, CUDA, CUDNN, Intel MKL,
various distributed filesystem such as HDFS/Amazon S3/...

  First copy [make/config.mk](../make/config.mk) to the project root, then
  modify the according flags.
