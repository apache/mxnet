# Convert MXNet models into Apple CoreML format.

This tool helps convert MXNet models into [Apple CoreML](https://developer.apple.com/documentation/coreml) format which can then be run on Apple devices.

## Installation
In order to use this tool you need to have these:
* MacOS - 10.11 (El Capitan) or higher (for running inferences on the converted model MacOS 10.13 or higher (for phones: iOS 11 or above) is needed)
* python 2.7
* mxnet-to-coreml tool: 

```bash
pip install mxnet-to-coreml
```

## How to use
Let's say you want to use your MXNet model in an iPhone App. For the purpose of this example, let's assume it is a squeezenet-v1.1 model.

1. Download the model into the directory where this converter resides. Squeezenet can be downloaded from [here](http://data.mxnet.io/models/imagenet/squeezenet/). The synset.txt file which contains all the class-labels and can be downloaded from [here](http://data.mxnet.io/models/imagenet/synset.txt).
2. Run this command:

  ```bash
mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,227,227"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels synset.txt --output-file="squeezenetv11.mlmodel"
```

  The above command will save the converted model in CoreML format to file squeezenet-v11.mlmodel. Internally, the model is first loaded by MXNet recreating the entire symbolic graph in memory. The converter walks through this symbolic graph converting each operator into its CoreML equivalent. Some of the supplied arguments to the converter are used by MXNet to generate the graph while others are used by CoreML either to pre-process the input (before passing it to the neural network) or to process the output of the neural network in a particular way.

  In the command above:

  * _model-prefix_: refers to the prefix of the file containing the MXNet model that needs to be converted (may include the directory path). E.g. for squeezenet model above the model files are squeezenet_v1.1-symbol.json and squeezenet_v1.1-0000.params and, therefore, model-prefix is "squeezenet_v1.1" (or "<directory-where-model-exists>/squeezenet_v1.1")
  * _epoch_: refers to the suffix of the MXNet model filename. For squeezenet model above, it'll be 0.
  * _input-shape_: refers to the input shape information in a JSON string format where the key is the name of the input variable (i.e. "data") and the value is the shape of that variable. If the model takes multiple inputs, input-shape for all of them need to be provided.
  * _mode_: refers to the coreml model mode. Can either be 'classifier', 'regressor' or None. In this case, we use 'classifier' since we want the resulting CoreML model to classify images into various categories.
  * _pre-processing-arguments_: In the Apple world, images have to be of type "Image". By providing image_input_names as "data", the converter will assume that the input variable "data" is of type "Image".
  * _class-labels_: refers to the name of the file which contains the classification labels (a.k.a. synset file).
  * _output-file_: the file where resulting CoreML model will be stored.

3. The generated ".mlmodel" file can directly be integrated into your app. For more instructions on how to do this, please see [Apple CoreML's tutorial](https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app).


### Providing class labels
You could provide a file containing class labels (as above) so that CoreML will return the category a given image belongs to. The file should have a label per line and labels can have any special characters. The line number of the label in the file should correspond with the index of softmax output. E.g.

```bash
mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,227,227"}' --mode=classifier --class-labels synset.txt --output-file="squeezenetv11.mlmodel"
```

### Adding a pre-processing layer to CoreML model.
You could ask CoreML to pre-process the images before passing them through the model. The following command provides image re-centering parameters for red, blue and green channel.

```bash
mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,224,224"}' --pre-processing-arguments='{"red_bias":127,"blue_bias":117,"green_bias":103}' --output-file="squeezenet_v11.mlmodel"
```

If you are building an app for a model that takes "Image" as an input, you will have to provide image_input_names as pre-processing arguments. This tells CoreML that a particular input variable is of type Image. E.g.:

```bash
mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,224,224"}' --pre-processing-arguments='{"red_bias":127,"blue_bias":117,"green_bias":103,"image_input_names":"data"}' --output-file="squeezenet_v11.mlmodel"
```

## Currently supported
### Layers
List of MXNet layers that can be converted into their CoreML equivalent:

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

### Models
Any MXNet model that uses the above operators can be converted easily. For instance, the following standard models can be converted:

1. [Inception-BN](http://data.mxnet.io/models/imagenet/inception-bn/)

```bash
mxnet_coreml_converter.py --model-prefix='Inception-BN' --epoch=126 --input-shape='{"data":"3,224,224"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels synset.txt --output-file="InceptionBN.mlmodel"
```

2. [NiN](http://data.dmlc.ml/models/imagenet/nin/)

```bash
mxnet_coreml_converter.py --model-prefix='nin' --epoch=0 --input-shape='{"data":"3,224,224"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels synset.txt --output-file="nin.mlmodel"
```

3. [Resnet](http://data.mxnet.io/models/imagenet/resnet/)

```bash
mxnet_coreml_converter.py --model-prefix='resnet-50' --epoch=0 --input-shape='{"data":"3,224,224"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels synset.txt --output-file="resnet50.mlmodel"
```

4. [Squeezenet](http://data.mxnet.io/models/imagenet/squeezenet/)

```bash
mxnet_coreml_converter.py --model-prefix='squeezenet_v1.1' --epoch=0 --input-shape='{"data":"3,227,227"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels synset.txt --output-file="squeezenetv11.mlmodel"
```

5. [Vgg](http://data.mxnet.io/models/imagenet/vgg/)

```bash
mxnet_coreml_converter.py --model-prefix='vgg16' --epoch=0 --input-shape='{"data":"3,224,224"}' --mode=classifier --pre-processing-arguments='{"image_input_names":"data"}' --class-labels synset.txt --output-file="vgg16.mlmodel"
```

## Known issues
* [Inception-V3](http://data.mxnet.io/models/imagenet/inception-v3.tar.gz) model can be converted into CoreML format but is unable to run on Xcode.
