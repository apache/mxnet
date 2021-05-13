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

# Exporting to ONNX format

[Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx) provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. In the MXNet 1.9 release, the MXNet-to-ONNX export module (mx2onnx) has received a major update with new features such as dynamic input shapes and better operator and model coverages. Please visit the [ONNX Export Support for MXNet](https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#onnx-export-support-for-mxnet) page for more information.

In this tutorial, we will learn how to use the mx2onnx exporter on pre-trained models.

## Prerequisites

To run the tutorial we will need to have installed the following python modules:
- [MXNet >= 1.9.0](/get_started) _OR_ an earlier MXNet version + [the mx2onnx wheel](https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#installation)
- [onnx >= 1.7.0](https://github.com/onnx/onnx#installation)

*Note:* The latest mx2onnx exporting module is tested with ONNX op set version 12 or later, which corresponds to ONNX version 1.7 or later. Use of ealier ONNX versions may still work on some simple models, but again this is not tested.


```python
import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
```

## Download a model from the MXNet model zoo

We can download a pre-trained ResNet-18 [ImageNet](http://www.image-net.org/) model from the [MXNet Model Zoo](/api/python/docs/api/gluon/model_zoo/index.html).
We will also download a synset file to match the labels.

```python
# Download pre-trained resnet model - json and params by running following code.
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
 mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]
```

## MXNet to ONNX exporter (mx2onnx) API

Now let's check MXNet's `export_model` API. 

```python
help(mx.onnx.export_model)
```

Output:

```text
Help on function export_model in module mxnet.onnx.mx2onnx._export_model:

export_model(sym, params, in_shapes=None, in_types=<class 'numpy.float32'>, onnx_file_path='model.onnx', verbose=False, dynamic=False, dynamic_input_shapes=None, run_shape_inference=False, input_type=None, input_shape=None)
    Exports the MXNet model file, passed as a parameter, into ONNX model.
    Accepts both symbol,parameter objects as well as json and params filepaths as input.
    Operator support and coverage -
    https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#operator-support-matrix
    
    Parameters
    ----------
    sym : str or symbol object
        Path to the json file or Symbol object
    params : str or dict or list of dict
        str - Path to the params file
        dict - params dictionary (Including both arg_params and aux_params)
        list - list of length 2 that contains arg_params and aux_params
    in_shapes : List of tuple
        Input shape of the model e.g [(1,3,224,224)]
    in_types : data type or list of data types
        Input data type e.g. np.float32, or [np.float32, np.int32]
    onnx_file_path : str
        Path where to save the generated onnx file
    verbose : Boolean
        If True will print logs of the model conversion
    dynamic: Boolean
        If True will allow for dynamic input shapes to the model
    dynamic_input_shapes: list of tuple
        Specifies the dynamic input_shapes. If None then all dimensions are set to None
    run_shape_inference : Boolean
        If True will run shape inference on the model
    input_type : data type or list of data types
        This is the old name of in_types. We keep this parameter name for backward compatibility
    input_shape : List of tuple
        This is the old name of in_shapes. We keep this parameter name for backward compatibility
    
    Returns
    -------
    onnx_file_path : str
        Onnx file path
    
    Notes
    -----
    This method is available when you ``import mxnet.onnx``
```

The `export_model` API can accept a MXNet model in one of the following ways.

1. MXNet's exported json and params files:
    * This is useful if we have pre-trained models and we want to convert them to ONNX format.
2. MXNet sym, params objects:
    * This is useful if we are training a model. At the end of training, we just need to invoke the `export_model` function and provide the sym and params objects as inputs to save the model in ONNX format. The params can be either a single object that contains both argument and auxiliary parameters, or a list that includes arg_parmas and aux_params objects

Since we have downloaded pre-trained model files, we will use the `export_model` API by passing in the paths of the symbol and params files.

## Use mx2onnx to eport the model

We will use the downloaded pre-trained model files (sym, params) and define a few more parameters.

```python
# Downloaded input symbol and params files
sym = './resnet-18-symbol.json'
params = './resnet-18-0000.params'

# Standard Imagenet input - 3 channels, 224 * 224
in_shapes = [(1, 3, 224, 224)]
in_types = [np.float32]

# Path of the output file
onnx_file = './mxnet_exported_resnet18.onnx'
```

We have defined the input parameters required for the `export_model` API. Now, we are ready to covert the MXNet model into ONNX format.

```python
# Invoke export model API. It returns path of the converted onnx model
converted_model_path = mx.onnx.export_model(sym, params, in_shapes, in_types, onnx_file)
```

This API returns the path of the converted model which you can later use to run inference with or import the model into other frameworks. Please refer to [mx2onnx](https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#apis) for more details about the API.

## Dynamic input shapes
The mx2onnx module also supports dynamic input shapes. We can set `dynamic=True` to turn it on. Note that even with dynamic shapes, a set of static input shapes still need to be specified in `in_shapes`; on top of that, we'll also need to specify which dimensions of the input shapes are dynamic in `dynamic_input_shapes`. We can simply set the dynamic dimensions as `None`, e.g. `(1, 3, None, None)`, or use strings in place of the `None`'s for better understandability in the exported onnx graph, e.g. `(1, 3, 'Height', 'Width')`

```python
# The first input dimension will be dynamic in this case
dynamic_input_shapes = [(None, 3, 224, 224)]
converted_model_path = mx.onnx.export_model(sym, params, in_shapes, in_types, onnx_file,
                                            dynamic=True, dynamic_input_shapes=dynamic_input_shapes)
```

## Validate the exported ONNX model

Now that we have the converted model, we can validate its correctness with the ONNX checker tool.

```python
from onnx import checker
import onnx

# Load the ONNX model
model_proto = onnx.load_model(converted_model_path)

# Check if the converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)
```

Now that the model passes the check (hopefully :)), we can run it with inference frameworks or import it into other deep learning frameworks!

## Simplify the exported ONNX model

Okay, we already have the exporeted ONNX model now, but it may not be the end of the story. Due to differences in MXNet's and ONNX's operator specifications, sometimes helper operartors/nodes will need to be created to help construct the ONNX graph from the MXNet blueprint. In that sense, we recommend our users to checkout [onnx-simplifier](https://github.com/daquexian/onnx-simplifier), which can greatly simply the exported ONNX model by techniques such as constant folding, operator fussion and more.
