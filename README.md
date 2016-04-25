<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnet2.png width=135/> *for Deep Learning*
=====

[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Build Status](http://ci.dmlc.ml/buildStatus/icon?job=mxnet)](http://ci.dmlc.ml/job/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](http://mxnet.readthedocs.org/en/latest/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![todofy badge](https://todofy.org/b/dmlc/mxnet)](https://todofy.org/r/dmlc/mxnet)

![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)

MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** the [flavours](http://mxnet.readthedocs.org/en/latest/program_model.html) of symbolic
programming and imperative programming to ***maximize*** efficiency and productivity.
In its core, a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
The library is portable and lightweight, and it scales to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of
[blue prints and guidelines](http://mxnet.readthedocs.org/en/latest/#open-source-design-notes) for building
deep learning system, and interesting insights of DL systems for hackers.

[![Join the chat at https://gitter.im/dmlc/mxnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/mxnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

What's New
----------
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [Embedding Torch layers and functions in MXNet](http://mxnet.readthedocs.org/en/latest/how_to/torch.html)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)
](https://github.com/dmlc/mxnet.js/)
* [Design Note: Design Efficient Deep Learning Data Loading Module](http://mxnet.readthedocs.org/en/latest/system/note_data_loading.htmll)
* [MXNet on Mobile Device](http://mxnet.readthedocs.org/en/latest/how_to/smart_device.html)
* [Distributed Training](http://mxnet.readthedocs.org/en/latest/how_to/multi_devices.html)
* [Guide to Creating New Operators (Layers)](http://mxnet.readthedocs.org/en/latest/how_to/new_op.html)
* [Amalgamation and Go Binding for Predictors](https://github.com/jdeng/gomxnet/)
* [Training Deep Net on 14 Million Images on A Single Machine](http://mxnet.readthedocs.org/en/latest/tutorials/imagenet_full.html)

Contents
--------
* [Documentation and Tutorials](http://mxnet.readthedocs.org/en/latest/)
* [Design Notes](http://mxnet.readthedocs.org/en/latest/system/index.html)
* [Code Examples](example)
* [Installation](http://mxnet.readthedocs.org/en/latest/how_to/build.html)
* [Pretrained Models](https://github.com/dmlc/mxnet-model-gallery)
* [Contribute to MXNet](http://mxnet.readthedocs.org/en/latest/how_to/contribute.html)
* [Frequent Asked Questions](http://mxnet.readthedocs.org/en/latest/how_to/faq.html)

Features
--------
* Design notes providing useful insights that can re-used by other DL projects
* Flexible configuration for arbitrary computation graph
* Mix and match good flavours of programming to maximize flexibility and efficiency
* Lightweight, memory efficient and portable to smart devices
* Scales up to multi GPUs and distributed setting with auto parallelism
* Support for python, R, C++ and Julia
* Cloud-friendly and directly compatible with S3, HDFS, and Azure

Ask Questions
-------------
* Please use [mxnet/issues](https://github.com/dmlc/mxnet/issues) for how to use mxnet and reporting bugs 

License
-------
© Contributors, 2015. Licensed under an [Apache-2.0](https://github.com/dmlc/mxnet/blob/master/LICENSE) license.

Reference Paper
---------------

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015

History
-------
MXNet is initiated and designed in collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva) and [purine2](https://github.com/purine/purine2). The project reflects what we have learnt from the past projects. It combines important flavours of the existing projects for efficiency, flexibility and memory efficiency.
