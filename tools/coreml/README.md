# Convert MXNet models into Apple CoreML format.

This tool helps convert MXNet models into [Apple CoreML](https://developer.apple.com/documentation/coreml) format which can then be run on Apple devices.

## Installation
In order to use this tool you need to have these installed:
* MacOS - High Sierra 10.13
* Xcode 9
* coremltools 0.5.0 or greater (pip install coremltools)
* mxnet 0.10.0 or greater. [Installation instructions](http://mxnet.io/get_started/install.html).
* yaml (pip install pyyaml)
* python 2.7

## How to use
Let's say you want to use your MXNet model in an iPhone App. For the purpose of this example, let's say you want to use squeezenet-v1.1.

1. Download the model into the directory where this converter resides. Squeezenet can be downloaded from [here](http://data.mxnet.io/models/imagenet/squeezenet/).
2. Run this command:

  ```bash
python mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,227,227"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels classLabels.txt --output-file="squeezenetv11.mlmodel"
```

  The above command will save the converted model into squeezenet-v11.mlmodel in CoreML format. Internally MXNet first loads the model and then we walk through the entire symbolic graph converting each operator into its CoreML equivalent. Some of the parameters are used by MXNet in order to load and generate the symbolic graph in memory while others are used by CoreML either to pre-process the input before the going through the neural network or to process the output in a particular way. 

  In the command above:

  * _model-prefix_: refers to the MXNet model prefix (may include the directory path).
  * _epoch_: refers to the suffix of the MXNet model file.
  * _input-shape_: refers to the input shape information in a JSON string format where the key is the name of the input variable (="data") and the value is the shape of that variable. If the model takes multiple inputs, input-shape for all of them need to be provided.
  * _mode_: refers to the coreml model mode. Can either be 'classifier', 'regressor' or None. In this case, we use 'classifier' since we want the resulting CoreML model to classify images into various categories.
  * _pre-processing-arguments_: In the Apple world images have to be of type Image. By providing image_input_names as "data", we are saying that the input variable "data" is of type Image.
  * _class-labels_: refers to the name of the file which contains the classification labels (a.k.a. synset file).
output-file: the file where the CoreML model will be dumped.

3. The generated ".mlmodel" file can directly be integrated into your app. For more instructions on how to do this, please see [Apple CoreML's tutorial](https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app).


### Providing class labels
You could provide a file containing class labels (as above) so that CoreML will return the predicted category the image belongs to. The file should have a label per line and labels can have any special characters. The line number of the label in the file should correspond with the index of softmax output. E.g.

```bash
python mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,227,227"}' --mode=classifier --class-labels classLabels.txt --output-file="squeezenetv11.mlmodel"
```

### Providing label names
You may have to provide the label names of the MXNet model's outputs. For example, if you try to convert [vgg16](http://data.mxnet.io/models/imagenet/vgg/), you may have to provide label-name as "prob_label". By default "softmax_label" is assumed.

```bash
python mxnet_coreml_converter.py --model-prefix='vgg16' --epoch=0 --input-shape='{"data":"3,224,224"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels classLabels.txt --output-file="vgg16.mlmodel" --label-names="prob_label"
```
 
### Adding a pre-processing to CoreML model.
You could ask CoreML to pre-process the images before passing them through the model.

```bash
python mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,224,224"}' --pre-processing-arguments='{"red_bias":127,"blue_bias":117,"green_bias":103}' --output-file="squeezenet_v11.mlmodel"
```

If you are building an app for a model that takes image as an input, you will have to provide image_input_names as pre-processing arguments. This tells CoreML that a particular input variable is of type Image. E.g.:
 
```bash
python mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,224,224"}' --pre-processing-arguments='{"red_bias":127,"blue_bias":117,"green_bias":103,"image_input_names":"data"}' --output-file="squeezenet_v11.mlmodel"
```

## Currently supported
### Models
This is a (growing) list of standard MXNet models that can be successfully converted using the converter. This means that any other model that uses similar operators as these models can also be successfully converted.

1. Inception: [Inception-BN](http://data.mxnet.io/models/imagenet/inception-bn/), [Inception-V3](http://data.mxnet.io/models/imagenet/inception-v3.tar.gz)
2. [NiN](http://data.dmlc.ml/models/imagenet/nin/)
2. [Resnet](http://data.mxnet.io/models/imagenet/resnet/)
3. [Squeezenet](http://data.mxnet.io/models/imagenet/squeezenet/)
4. [Vgg](http://data.mxnet.io/models/imagenet/vgg/)

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
Currently there are no known issues.

## This tool has been tested on environment with:
* MacOS - High Sierra 10.13 Beta.
* Xcode 9 beta 5.
