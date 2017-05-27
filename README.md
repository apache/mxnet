<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnet2.png width=135/> *for Deep Learning*
=====

[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](http://mxnet.io/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)

MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of
[blue prints and guidelines](http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts) for building
deep learning systems, and interesting insights of DL systems for hackers.

[![Join the chat at https://gitter.im/dmlc/mxnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/mxnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

What's New
----------
* [Version 0.10.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.10.0) - MXNet 0.10.0 Release.
* [Version 0.9.3 Release](./docs/architecture/release_note_0_9.md) - First 0.9 official release.
* [Version 0.9.1 Release (NNVM refactor)](./docs/architecture/release_note_0_9.md) - NNVM branch is merged into master now. An official release will be made soon.
* [Version 0.8.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.8.0)
* [Updated Image Classification with new Pre-trained Models](./example/image-classification)
* [Python Notebooks for How to Use MXNet](https://github.com/dmlc/mxnet-notebooks)
* [MKLDNN for Faster CPU Performance](./MKL_README.md)
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [Embedding Torch layers and functions in MXNet](http://mxnet.io/how_to/torch.html)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)
](https://github.com/dmlc/mxnet.js/)
* [Design Note: Design Efficient Deep Learning Data Loading Module](http://mxnet.io/architecture/note_data_loading.html)
* [MXNet on Mobile Device](http://mxnet.io/how_to/smart_device.html)
* [Distributed Training](http://mxnet.io/how_to/multi_devices.html)
* [Guide to Creating New Operators (Layers)](http://mxnet.io/how_to/new_op.html)
* [Go binding for inference](https://github.com/songtianyi/go-mxnet-predictor)
* [Amalgamation and Go Binding for Predictors](https://github.com/jdeng/gomxnet/) - Outdated
* [Training Deep Net on 14 Million Images on A Single Machine](http://mxnet.io/tutorials/computer_vision/imagenet_full.html)

Contents
--------
* [Documentation and Tutorials](http://mxnet.io/)
* [Design Notes](http://mxnet.io/architecture/index.html)
* [Code Examples](https://github.com/dmlc/mxnet/tree/master/example)
* [Installation](http://mxnet.io/get_started/install.html)
* [Pretrained Models](https://github.com/dmlc/mxnet-model-gallery)
* [Contribute to MXNet](http://mxnet.io/community/contribute.html)
* [Frequent Asked Questions](http://mxnet.io/how_to/faq.html)

Features
--------
* Design notes providing useful insights that can re-used by other DL projects
* Flexible configuration for arbitrary computation graph
* Mix and match imperative and symbolic programming to maximize flexibility and efficiency
* Lightweight, memory efficient and portable to smart devices
* Scales up to multi GPUs and distributed setting with auto parallelism
* Support for Python, R, Scala, C++ and Julia
* Cloud-friendly and directly compatible with S3, HDFS, and Azure

Ask Questions
-------------
* Please use [mxnet/issues](https://github.com/dmlc/mxnet/issues) for how to use mxnet and reporting bugs

License
-------
© Contributors, 2015-2017. Licensed under an [Apache-2.0](https://github.com/dmlc/mxnet/blob/master/LICENSE) license.

Reference Paper
---------------

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015

History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.
