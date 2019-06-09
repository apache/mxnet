# MXNet C++ Package Inference Workflow Examples

## Building C++ Inference examples

The examples in this folder demonstrate the **inference** workflow.
To build examples use following commands:

-  Release: **make all**
-  Debug: **make debug all**


## Examples demonstrating inference workflow

This directory contains following examples. In order to run the examples, ensure that the path to the MXNet shared library is added to the OS specific environment variable viz. **LD\_LIBRARY\_PATH** for Linux, Mac and Ubuntu OS and **PATH** for Windows OS.

### [inception_inference.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/inception_inference.cpp>)

This example demonstrates image classification workflow with pre-trained models using MXNet C++ API. The command line parameters the example can accept are as shown below:

```
./inception_inference --help
Usage:
inception_inference --symbol <model symbol file in json format>
                    --params <model params file>
					--image <path to the image used for prediction
					--synset file containing labels for prediction
					[--input_shape <dimensions of input image e.g "3 224 224"]
					[--mean file containing mean image for normalizing the input image
					[--gpu] Specify this option if workflow needs to be run in gpu context
```
The model json and param file and synset files are required to run this example.  The sample command line is as follows:

```

./inception_inference --symbol "./model/Inception-BN-symbol.json" --params "./model/Inception-BN-0126.params" --synset "./model/synset.txt" --mean "./model/mean_224.nd" --image "./model/dog.jpg"
```
Alternatively, The script [unit_test_inception_inference.sh](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/unit_test_inception_inference.sh>) downloads the pre-trained **Inception** model and a test image. The users can invoke this script as follows:

```
./unit_test_inception_inference.sh
```
