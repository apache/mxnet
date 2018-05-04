# ONNX-MXNet API

## Overview

[ONNX](https://onnx.ai/) is an open format to represent deep learning models. With ONNX as an intermediate representation, it is easier to move models between state-of-the-art tools and frameworks for training and inference.

The `mxnet.contrib.onnx` package refers to the APIs and interfaces that implement ONNX model format support for Apache MXNet.

With ONNX format support for MXNet, developers can build and train models with a [variety of deep learning frameworks](http://onnx.ai/supported-tools), and import these models into MXNet to run them for inference and training using MXNet’s highly optimized engine.

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
```

### Installation Instructions
- To use this module developers need to **install ONNX**, which requires protobuf compiler to be installed separately. Please follow the [instructions to install ONNX and its dependencies](https://github.com/onnx/onnx#installation). Once installed, you can go through the tutorials on how to use this module.


This document describes all the ONNX-MXNet APIs.

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.contrib.onnx.import_model
```

## ONNX Tutorials

```eval_rst
.. toctree::
   :maxdepth: 1
   
   /tutorials/onnx/super_resolution.md
   /tutorials/onnx/inference_on_onnx_model.md
   /tutorials/onnx/fine_tuning_gluon.md
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.contrib.onnx
    :members: import_model 

```

<script>auto_index("api-reference");</script>