MXNet
=====

[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](http://mxnet.readthedocs.org/en/latest/)
[![GitHub Stats](https://img.shields.io/badge/github-stats-ff5500.svg)](http://githubstats.com/dmlc/mxnet)
[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)]()


Contents
--------
* [Documentation](http://mxnet.readthedocs.org/en/latest/)
* [Build Instruction](doc/build.md)
* [Features](#features)
* [License](#license)

Features
--------
* Lightweight: small but sharp knife
  - mxnet contains concise implementation of state-of-art deep learning models
  - The project maintains a minimum dependency that makes it portable and easy to build
* Scalable and beyond
  - The package scales to multiple GPUs already with an easy to use kvstore.
  - The same code can be ported to distributed version when the distributed kvstore is ready.
* Multi-GPU NDArray/Tensor API with auto parallelization
  - The package supports a flexible ndarray interface that runs on both CPU and GPU, more importantly
    automatically parallelize the computation for you.
* Language agnostic
  - The package currently support C++ and python, with a clean C API.
  - This makes the package being easily portable to other languages and platforms.
* Cloud friendly
  - MXNet is ready to work with cloud storages including S3, HDFS, AZure for data source and model saving.
  - This means you do can put data on S3 directly using it to train your deep model.
* Easy extensibility with no requirement on GPU programming
  - The package can be extended in several scopes, including python, c++.
  - In all these levels, developers can write numpy style expressions, either via python
    or [mshadow expression template](https://github.com/dmlc/mshadow).
  - It brings concise and readable code, with performance matching hand crafted kernels

Bug Reporting
-------------
* For reporting bugs please use the [mxnet/issues](https://github.com/dmlc/mxnet/issues) page.

Contributing to MXNet
---------------------
MXNet has been developed and used by a group of active community members.
Everyone is more than welcome to contribute. It is a way to make the project better and more accessible to more users.
* Please add your name to [CONTRIBUTORS.md](CONTRIBUTORS.md) after your patch has been merged.

License
-------
© Contributors, 2015. Licensed under an [Apache-2.0](https://github.com/dmlc/mxnet/blob/master/LICENSE) license.
