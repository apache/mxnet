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

[Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx) provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.

In this tutorial, we will show how you can save MXNet models to the ONNX format.

MXNet-ONNX operators coverage and features are updated regularly. Visit the [ONNX operator coverage](https://cwiki.apache.org/confluence/display/MXNET/ONNX+Operator+Coverage) page for the latest information.

In this tutorial, we will learn how to use MXNet to ONNX exporter on pre-trained models.

## Prerequisites

To run the tutorial you will need to have installed the following python modules:
- [MXNet >= 2.0.0](https://mxnet.apache.org/get_started)
- [onnx]( https://github.com/onnx/onnx#user-content-installation) v1.7 & v1.8 (follow the install guide)

*Note:* MXNet-ONNX importer and exporter follows version 12 & 13 of ONNX operator set which comes with ONNX v1.7 & v1.8.


```{.python .input}
import mxnet as mx
from mxnet import initializer as init, np, onnx as mxnet_onnx
from mxnet.gluon import nn
import logging
logging.basicConfig(level=logging.INFO)
```

## Create a model from the MXNet Gluon

Let's build a concise model with [MXNet gluon](../../../api/gluon/index.rst) package. The model is multilayer perceptrons with two fully-connected layers. The first one is our hidden layer, which contains 256 hidden units and applies ReLU activation function. The second is our output layer. 

```{.python .input}
net = nn.HybridSequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
```

Then we initialize the model and export it into symbol file and parameter file. 

```{.python .input}
net.initialize(init.Normal(sigma=0.01))
net.hybridize()
input = np.ones(shape=(50,), dtype=np.float32)
output = net(input)
net.export("mlp")
```

Now, we have exported the model symbol, params file on the disk.

## MXNet to ONNX exporter API

Let us describe the MXNet's `export_model` API.

```{.python .input}
help(mxnet_onnx.export_model)
```

Output:

```text
Help on function export_model in module mxnet.contrib.onnx.mx2onnx.export_model:

export_model(sym, params, input_shape, input_type=<type 'numpy.float32'>, onnx_file_path=u'model.onnx', verbose=False)
    Exports the MXNet model file, passed as a parameter, into ONNX model.
    Accepts both symbol,parameter objects as well as json and params filepaths as input.
    Operator support and coverage - https://cwiki.apache.org/confluence/display/MXNET/MXNet-ONNX+Integration

    Parameters
    ----------
    sym : str or symbol object
        Path to the json file or Symbol object
    params : str or symbol object
        Path to the params file or params dictionary. (Including both arg_params and aux_params)
    input_shape : List of tuple
        Input shape of the model e.g [(1,3,224,224)]
    input_type : data type
        Input data type e.g. np.float32
    onnx_file_path : str
        Path where to save the generated onnx file
    verbose : Boolean
        If true will print logs of the model conversion

    Returns
    -------
    onnx_file_path : str
        Onnx file path
```

`export_model` API can accept the MXNet model in one of the following two ways.

1. MXNet sym, params objects:
    * This is useful if we are training a model. At the end of training, we just need to invoke the `export_model` function and provide sym and params objects as inputs with other attributes to save the model in ONNX format.
2. MXNet's exported json and params files:
    * This is useful if we have pre-trained models and we want to convert them to ONNX format.

Since we have downloaded pre-trained model files, we will use the `export_model` API by passing the path for symbol and params files.

## How to use MXNet to ONNX exporter API

We will use the downloaded pre-trained model files (sym, params) and define input variables.

```{.python .input}
# The input symbol and params files
sym = './mlp-symbol.json'
params = './mlp-0000.params'

# Standard Imagenet input - 3 channels, 224*224
input_shape = (50,)

# Path of the output file
onnx_file = './mxnet_exported_mlp.onnx'
```

We have defined the input parameters required for the `export_model` API. Now, we are ready to covert the MXNet model into ONNX format.

```{.python .input}
# Invoke export model API. It returns path of the converted onnx model
converted_model_path = mxnet_onnx.export_model(sym, params, [input_shape], [np.float32], onnx_file)
```

This API returns path of the converted model which you can later use to import the model into other frameworks.

## Check validity of ONNX model

Now we can check validity of the converted ONNX model by using ONNX checker tool. The tool will validate the model by checking if the content contains valid protobuf:

```{.python .input}
from onnx import checker
import onnx

# Load onnx model
model_proto = onnx.load_model(converted_model_path)

# Check if converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)
```

If the converted protobuf format doesn't qualify to ONNX proto specifications, the checker will throw errors, but in this case it successfully passes.

This method confirms exported model protobuf is valid. Now, the model is ready to be imported in other frameworks for inference!
