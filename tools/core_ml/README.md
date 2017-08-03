# Convert MXNet models into Apple CoreML format.

This tool helps convert MXNet models into [Apple CoreML](https://developer.apple.com/documentation/coreml) format which can then be run on Apple devices.

## Installation
In order to use this tool you need to have these installed:
* mxnet 0.10.0. [Installation instructions](http://mxnet.io/get_started/install.html).
* python 2.7
* coremltools 0.4.0 (pip install coremltools)
* yaml (pip install pyyaml)

## How to use
Let's say you want to use your MXNet model in an iPhone App. For the purpose of this example, let's say you want to use squeezenet-v1.1.

1. Download the model into the directory where this converter resides. Squeezenet can be downloaded from [here](http://data.mxnet.io/models/imagenet/squeezenet/).
2. Run this command:
```bash
python mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,244,244"}' --output-file="squeezenet_v11.mlmodel"
```
The above command will save the converted model into squeezenet-v11.mlmodel in CoreML format.

3. This generated ".mlmodel" file can directly be integrated into your app. For more instructions on how to do this, please see [Apple CoreML's tutorial](https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app).


### Forced conversion
For some models there may not be a one-to-one correspondence with CoreML and the converter will fail if you are converting such models. If you understand the risks with the model conversion, you can provide a "force" flag to force the converter to convert. For instance for resnet models:

```bash
python mxnet_coreml_converter.py --model-prefix='resnet-50' --epoch=0 --input-shape='{"data":"3,244,244"}' --output-file="resnet-50.mlmodel" --force=True
```

### Providing class labels
TODO:
E.g. on providing synsets.

### Providing pre-processing arguments for CoreML model.
E.g. on providing pre-processing arguments.

## Currently supported
### Models
This is a (growing) list of standard MXNet models that can be successfully converted using the converter. This means that any other model that uses the similar operators as these models can also be successfully converted.
1. [Inception-BN](http://data.mxnet.io/models/imagenet/inception-bn/) (use force=True)
2. [Inception-V3](http://data.mxnet.io/models/imagenet/inception-v3.tar.gz).
3. [Network-In-Network](http://data.mxnet.io/models/imagenet/nin/)
4. [Squeezenet-V1.1](http://data.mxnet.io/models/imagenet/squeezenet/)
5. [Resnet](http://data.mxnet.io/models/imagenet/resnet/) (use force=True)
6. [Vgg](http://data.mxnet.io/models/imagenet/vgg/)

### Layers
1. Activation
2. Batchnorm
3. Concat
4. Convolution
5. Deconvolution
6. Dense
7. Elementwise
8. Flatten
9. Pooling
10. Reshape
11. Softmax
12. Transpose

## Known issues
These are list of known issues:
1. Deconvolution layer with padding results in incorrect CoreML predictions.

## This tool has been tested on environment with:
* MacOS - High Sierra 10.13 Beta
* Xcode 9 beta 2
