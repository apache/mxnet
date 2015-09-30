<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnet2.png width=135/> for Deep Learning
=====

[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](http://mxnet.readthedocs.org/en/latest/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to mix the [flavours](http://mxnet.readthedocs.org/en/latest/program_model.html) of
deep learning programs together to maximize the efficiency and your productivity.


What's New
----------
* [Note on Programming Models for Deep Learning](http://mxnet.readthedocs.org/en/latest/program_model.html)
* [Pretrained Inception BatchNorm Network](example/notebooks/predict-with-pretrained-model.ipynb)

Contents
--------
* [Documentation and Tutorials](http://mxnet.readthedocs.org/en/latest/)
* [Open Source Design Notes](http://mxnet.readthedocs.org/en/latest/#open-source-design-notes)
* [Code Examples](example)
* [Build Instruction](doc/build.md)
* [Features](#features)
* [License](#license)

Features
--------
* To Mix and Maximize
  - Mix all flavours of programming models to maximize flexibility and efficiency.
* Lightweight, scalable and memory efficient.
  - Minimum build dependency, scales to multi-GPUs with very low memory usage.
* Auto parallelization
  - Write numpy-style ndarray GPU programs, which will be automatically parallelized.
* Language agnostic
  - With support for python, c++, more to come.
* Cloud friendly
  - Directly load/save from S3, HDFS, AZure
* Easy extensibility
  - Extending no requirement on GPU programming.

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


History
-------
MXNet is initiated and designed in collaboration by authors from [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva) and [purine2](https://github.com/purine/purine2). The project reflects what we have learnt from the past projects. It combines important flavour of the existing projects, being efficient, flexible and memory efficient.
