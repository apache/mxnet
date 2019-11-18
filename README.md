<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<div align="center">
  <a href="https://mxnet.apache.org/"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png"></a><br>
</div>

Apache MXNet (incubating) for Deep Learning
=====
| Master         | Docs          | License  |
| :-------------:|:-------------:|:--------:|
[![CentOS CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-cpu/job/master/badge/icon?subject=build%20centos%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-cpu/job/master/) [![CentOS GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-gpu/job/master/badge/icon?subject=build%20centos%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-gpu/job/master/) [![Clang Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/clang/job/master/badge/icon?subject=build%20clang)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/clang/job/master/) <br> [![Edge Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/edge/job/master/badge/icon?subject=build%20edge)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/edge/job/master/) [![Miscellaneous Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/miscellaneous/job/master/badge/icon?subject=build%20miscellaneous)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/miscellaneous/job/master/) [![Sanity Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/sanity/job/master/badge/icon?subject=build%20sanity)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/sanity/job/master/) <br> [![Unix CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-cpu/job/master/badge/icon?subject=build%20unix%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-cpu/job/master/) [![Unix GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-gpu/job/master/badge/icon?subject=build%20unix%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-gpu/job/master/) [![Website Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/website/job/master/badge/icon?subject=build%20website)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/website/job/master/) <br> [![Windows CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-cpu/job/master/badge/icon?subject=build%20windows%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-cpu/job/master/) [![Windows GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-gpu/job/master/badge/icon?subject=build%20windows%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-gpu/job/master/) | [![Documentation Status](http://jenkins.mxnet-ci.amazon-ml.com/job/restricted-website-build/badge/icon)](https://mxnet.apache.org/) | [![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE) |

![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)

Apache MXNet (incubating) is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](https://mxnet.apache.org/api/architecture/program_model)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is more than a deep learning project. It is a collection of
[blue prints and guidelines](https://mxnet.apache.org/api/architecture/overview) for building
deep learning systems, and interesting insights of DL systems for hackers.

Ask Questions
-------------
* Please use [discuss.mxnet.io](https://discuss.mxnet.io/) for asking questions.
* Please use [mxnet/issues](https://github.com/apache/incubator-mxnet/issues) for reporting bugs.
* [Frequent Asked Questions](https://mxnet.apache.org/faq/faq.html)

How to Contribute
-----------------
* [Contribute to MXNet](https://mxnet.apache.org/community/contribute.html)

What's New
----------
* [Version 1.5.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.5.1) - MXNet 1.5.1 Patch Release.
* [Version 1.5.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.5.0) - MXNet 1.5.0 Release.
* [Version 1.4.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.4.1) - MXNet 1.4.1 Patch Release.
* [Version 1.4.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.4.0) - MXNet 1.4.0 Release.
* [Version 1.3.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.3.1) - MXNet 1.3.1 Patch Release.
* [Version 1.3.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.3.0) - MXNet 1.3.0 Release.
* [Version 1.2.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.2.0) - MXNet 1.2.0 Release.
* [Version 1.1.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.1.0) - MXNet 1.1.0 Release.
* [Version 1.0.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.0.0) - MXNet 1.0.0 Release.
* [Version 0.12.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.1) - MXNet 0.12.1 Patch Release.
* [Version 0.12.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.0) - MXNet 0.12.0 Release.
* [Version 0.11.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.11.0) - MXNet 0.11.0 Release.
* [Apache Incubator](http://incubator.apache.org/projects/mxnet.html) - We are now an Apache Incubator project.
* [Version 0.10.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.10.0) - MXNet 0.10.0 Release.
* [Version 0.9.3 Release](./docs/architecture/release_note_0_9.md) - First 0.9 official release.
* [Version 0.9.1 Release (NNVM refactor)](./docs/architecture/release_note_0_9.md) - NNVM branch is merged into master now. An official release will be made soon.
* [Version 0.8.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.8.0)
* [Updated Image Classification with new Pre-trained Models](./example/image-classification)
* [Notebooks How to Use MXNet](https://github.com/d2l-ai/d2l-en)
* [MKLDNN for Faster CPU Performance](docs/python_docs/python/tutorials/performance/backend/mkldnn/mkldnn_readme.md)
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)](https://github.com/dmlc/mxnet.js/)
* [Guide to Creating New Operators (Layers)](https://mxnet.apache.org/api/faq/new_op)
* [Go binding for inference](https://github.com/songtianyi/go-mxnet-predictor)
* [Amalgamation and Go Binding for Predictors](https://github.com/jdeng/gomxnet/) - Outdated
* [Large Scale Image Classification](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification)

Contents
--------
* [Website](https://mxnet.apache.org)
* [Documentation](https://mxnet.apache.org/api)
* [Blog](https://mxnet.apache.org/blog)
* [Code Examples](https://github.com/apache/incubator-mxnet/tree/master/example)
* [Installation](https://mxnet.apache.org/get_started)
* [Features](https://mxnet.apache.org/features)
* [Ecosystem](https://mxnet.apache.org/ecosystem)

Features
--------
* Design notes providing useful insights that can re-used by other DL projects
* Flexible configuration for arbitrary computation graph
* Mix and match imperative and symbolic programming to maximize flexibility and efficiency
* Lightweight, memory efficient and portable to smart devices
* Scales up to multi GPUs and distributed setting with auto parallelism
* Support for [Python](https://mxnet.apache.org/api/python), [Scala](https://mxnet.apache.org/api/scala), [C++](https://mxnet.apache.org/api/cpp), [Java](https://mxnet.apache.org/api/java), [Clojure](https://mxnet.apache.org/api/clojure), [R](https://mxnet.apache.org/api/r), [Go](https://github.com/jdeng/gomxnet/), [Javascript](https://github.com/dmlc/mxnet.js/), [Perl](https://mxnet.apache.org/api/perl), [Matlab](https://github.com/apache/incubator-mxnet/tree/master/matlab), and [Julia](https://mxnet.apache.org/api/julia)
* Cloud-friendly and directly compatible with AWS S3, AWS Deep Learning AMI, AWS SageMaker, HDFS, and Azure

License
-------
Licensed under an [Apache-2.0](https://github.com/apache/incubator-mxnet/blob/master/LICENSE) license.

Reference Paper
---------------

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015

History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.
