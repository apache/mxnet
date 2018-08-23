
# Exporting MXNet model to ONNX format

[Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx) provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.

In this tutorial, we will show how you can save MXNet models to the ONNX format.

MXNet-ONNX operators coverage and features are updated regularly. Visit the [ONNX operator coverage](https://cwiki.apache.org/confluence/display/MXNET/ONNX+Operator+Coverage) page for the latest information.

In this tutorial we will learn how to use MXNet to ONNX exporter on pre-trained models.

## Prerequisites

To run the tutorial you will need to have installed the following python modules:
- [MXNet == 1.3.0](http://mxnet.incubator.apache.org/install/index.html)
- [onnx](https://github.com/onnx/onnx) v1.2.1 (follow the install guide)

*Note:* MXNet ONNX importer and exporter follows version 7 of ONNX operator set which comes with ONNX v1.2.1.


```python
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)
```

## Downloading a model from the MXNet model zoo

We download a pre-trained model, in this case ResNet-18 model, trained on [ImageNet](http://www.image-net.org/) from the [MXNet Model Zoo](http://data.mxnet.io/models/imagenet/).
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

We can check MXNet's ONNX `export_model` API usage as follows: 

```python
help(onnx_mxnet.export_model)
```

From the above API description, we can see that the `export_model` API accepts two kinds of inputs:

1. MXNet sym, params objects:
    * This is useful if we are training a model. At the end of training, we just need to invoke the `export_model` function and provide sym and params objects as inputs with other attributes to save the model in ONNX format.
2. MXNet's exported json and params files:
    * This is useful if we have pre-trained models and we want to convert them to ONNX format.

In this tutorial, we will show second use case to convert pre-trained model to ONNX format:

## How to use MXNet to ONNX exporter API

We will use downloaded files and define input variables.

```python
# Downloaded input symbol and params files
sym = 'resnet-18-symbol.json'
params = 'resnet-18-0000.params'
# Standard Imagenet input - 3 channels, 224*224
input_shape = (1,3,224,224)
# Path of the output file
onnx_file = 'mxnet_exported_resnet50.onnx'
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
