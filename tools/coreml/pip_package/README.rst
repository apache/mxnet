MXNET -> CoreML Converter
=========================

`Apache MXNet <https://github.com/apache/incubator-mxnet>`_ (incubating) is a deep learning framework designed for both efficiency and flexibility. It allows you to mix `symbolic and imperative programming <http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts>`_ to maximize efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

`Core ML <http://developer.apple.com/documentation/coreml>`_ is an Apple framework which allows developers to simply and easily integrate machine learning (ML) models into apps running on Apple devices (including iOS, watchOS, macOS, and tvOS). Core ML introduces a public file format (.mlmodel) for a broad set of ML methods including deep neural networks (both convolutional and recurrent), tree ensembles with boosting, and generalized linear models. Models in this format can be directly integrated into apps through Xcode.

This tool helps convert `MXNet models <https://github.com/apache/incubator-mxnet>`_ into `Apple CoreML <https://developer.apple.com/documentation/coreml>`_ format which can then be run on Apple devices. You can find more information about this tool on our `github <https://github.com/apache/incubator-mxnet/tree/master/tools/coreml>`_ page.

Prerequisites
-------------
This package can only be installed on MacOS X since it relies on Apple's CoreML SDK. It can be run on MacOS 10.11 or higher though for running inferences on the converted model MacOS 10.13 or higher is needed (or for phones, iOS 11 or above).

Installation
------------
The method for installing this tool follows the `standard python package installation steps <https://packaging.python.org/installing/>`_. Once you have set up a python environment, run::

  pip install mxnet-to-coreml

The package `documentation <https://github.com/apache/incubator-mxnet/tree/master/tools/coreml>`_ contains more details on how to use coremltools.

Dependencies
------------
This tool has the following dependencies:

* mxnet (0.10.0+)
* coremltools (0.5.1+)
* pyyaml (3.12+)

Sample Usage
------------

In order to convert, say a `Squeezenet model <http://data.mxnet.io/models/imagenet/squeezenet/>`_, with labels from `synset.txt <http://data.mxnet.io/models/imagenet/synset.txt>`_, execute this ::

  mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' \
  --epoch=0 --input-shape='{"data":"3,227,227"}' \
  --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' \
  --class-labels synset.txt --output-file="squeezenetv11.mlmodel"

More Information
----------------
* `On Github <https://github.com/apache/incubator-mxnet/tree/master/tools/coreml>`_
* `MXNet framework <https://github.com/apache/incubator-mxnet>`_
* `Apple CoreML <https://developer.apple.com/documentation/coreml>`_
