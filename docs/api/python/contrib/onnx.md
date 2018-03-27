# ONNX-MXNet API

## Overview

The `mxnet.contrib.onnx` package refers to the APIs and interfaces that implements ONNX model format support for Apache MXNet.

With ONNX format support for MXNet, developers can build and train models with PyTorch, CNTK, or Caffe2, and import these models into MXNet to run them for inference using MXNet’s highly optimized engine.

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
```

```eval_rst
.. note:: **Install ONNX** which needs protobuf compiler to be installed separately. Please **follow the instructions to install ONNX** - https://github.com/onnx/onnx.
```

This document describes the ONNX APIs in mxnet.

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
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.contrib.onnx
    :members: import_model 

```

<script>auto_index("api-reference");</script>