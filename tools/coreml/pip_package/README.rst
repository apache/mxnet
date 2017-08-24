MXNET -> CoreML Converter
=========================

This tool helps convert MXNet models into `Apple CoreML <https://developer.apple.com/documentation/coreml>`_ format which can then be run on Apple devices. Find more information `here <https://github.com/apache/incubator-mxnet/tree/master/tools/coreml>`_.

Prerequisites
-------------
This package can only be installed on MacOS X since it relies on Apple's CoreML SDK. This tool can be run on MacOS 10.12 or higher though for running inferences on the converted model MacOS 10.13 or higher is needed (or for phones, iOS 11 or above).

Installation
------------
To install::

  pip install mxnet-coreml-converter


Sample Usage
------------

In order to convert, say a `Squeezenet model <http://data.mxnet.io/models/imagenet/squeezenet/>`_, with labels from `synset.txt <http://data.mxnet.io/models/imagenet/synset.txt>`_, execute this ::

  mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' \
  --epoch=0 --input-shape='{"data":"3,227,227"}' \
  --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' \
  --class-labels synset.txt --output-file="squeezenetv11.mlmodel"
