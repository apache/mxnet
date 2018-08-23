
# Exporting MXNet model to ONNX format

[Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx) provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.

In this tutorial, we will show how you can save MXNet models to the ONNX format.

MXNet-ONNX operators coverage and features are updated regularly. Visit the [ONNX operator coverage](https://cwiki.apache.org/confluence/display/MXNET/ONNX+Operator+Coverage) page for the latest information.

In this tutorial, we will learn how to use MXNet to ONNX exporter on pre-trained models.

## Prerequisites

To run the tutorial you will need to have installed the following python modules:
- [MXNet >= 1.3.0](http://mxnet.incubator.apache.org/install/index.html)
- [onnx]( https://github.com/onnx/onnx#installation) v1.2.1 (follow the install guide)

*Note:* MXNet-ONNX importer and exporter follows version 7 of ONNX operator set which comes with ONNX v1.2.1.


```python
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)
```

## Downloading a model from the MXNet model zoo

We download the pre-trained ResNet-18 [ImageNet](http://www.image-net.org/) model from the [MXNet Model Zoo](http://data.mxnet.io/models/imagenet/).
We will also download synset file to match labels.

```python
# Download pre-trained resnet model - json and params by running following code.
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
 mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]
```

Now, we have downloaded ResNet-18 symbol, params and synset file on the disk.

## MXNet to ONNX exporter API

Let us describe the MXNet's `export_model` API. 

```python
help(onnx_mxnet.export_model)
```

```python
Help on function export_model in module mxnet.contrib.onnx.mx2onnx.export_model:

export_model(sym, params, input_shape, input_type=<type 'numpy.float32'>, onnx_file_path=u'model.onnx', verbose=False)
    Exports the MXNet model file, passed as a parameter, into ONNX model.
    Accepts both symbol,parameter objects as well as json and params filepaths as input.
    Operator support and coverage - https://cwiki.apache.org/confluence/display/MXNET/ONNX
    
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

```python
# Downloaded input symbol and params files
sym = './resnet-18-symbol.json'
params = './resnet-18-0000.params'

# Standard Imagenet input - 3 channels, 224*224
input_shape = (1,3,224,224)

# Path of the output file
onnx_file = './mxnet_exported_resnet50.onnx'
```

We have defined the input parameters required for the `export_model` API. Now, we are ready to covert the MXNet model into ONNX format.

```python
# Invoke export model API. It returns path of the converted onnx model
converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
```

This API returns path of the converted model which you can later use to import the model into other frameworks.

## Check validity of ONNX model

Now we can check validity of the converted ONNX model by using ONNX checker tool. The tool will validate the model by checking if the content contains valid protobuf:

```python
from onnx import checker
import onnx

# Load onnx model
model_proto = onnx.load(converted_model_path)

# Check if converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)
```

If the converted protobuf format doesn't qualify to ONNX proto specifications, the checker will throw errors, but in this case it successfully passes. 

This method confirms exported model protobuf is valid. Now, the model is ready to be imported in other frameworks for inference!
    
<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
