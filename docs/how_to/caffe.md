# How to Start for Caffe Users

This document go through topics for Caffe users to use MXNet.

- [Concepts difference between Caffe and MXNet]()
- [How to convert models trained in Caffe to MXNet]()
- [How to call Caffe opertor in MXNet]()

## Concepts difference between Caffe and MXNet

## How to convert models trained in Caffe to MXNet

### Build (Linux)

Either [Caffe's python package](http://caffe.berkeleyvision.org/installation.html) or [Google protobuf](https://developers.google.com/protocol-buffers/?hl=en) is required. The latter is often much easier to install:

1. We first install the protobuf compiler. If you compiled mxnet with `USE_DIST_KVSTORE = 1` then it is already built. Otherwise, install `protobuf-compiler` by your favor package manager, e.g. `sudo apt-get install protobuf-compiler` for ubuntu and `sudo yum install protobuf-compiler` for redhat/fedora.

2. Then install the protobuf's python binding. For example `sudo pip install protobuf`

Now we can build the tool by running `make` in the current directory.

### Build (Windows)

Note: this tool currently only works on python 2.

We must make sure that the installed python binding and protobuf compiler are using the same version of protobuf,
so we install the bindings first, and then install the corresponding compiler.

1. Install the protobuf bindings. At time of writing, the conda package manager has the most up to date version. Either run `conda install -c conda-forge protobuf` or `pip install protobuf`
2. Download the win32 build of protoc from [Protocol Buffers Releases](https://github.com/google/protobuf/releases). Make sure to download the version that corresponds to the version of the bindings. Extract to any location then add that location to your `PATH`
3. Run `make_win32.bat` to build the package


### How to use

Linux: Use `./run.sh model_name` to download and convert a model. E.g. `./run.sh vgg19`

Windows: Use `python convert_model.py prototxt caffemodel outputprefix`
For example: `python convert_model.py VGG_ILSVRC_16_layers_deploy.prototxt VGG_ILSVRC_16_layers.caffemodel vgg16`


### Note

* We have verified the results of VGG_16/VGG_19 model and BVLC_googlenet results from Caffe model zoo.
* The tool only supports single input and single output network.
* The tool can only work with the L2LayerParameter in Caffe.
* Caffe uses a convention for multi-strided pooling output shape inconsistent with MXNet
    * This importer doesn't handle this problem properly yet
    * And example of this failure is importing bvlc_Googlenet. The user needs to add padding to stride-2 pooling to make this work right now.

## How to call Caffe opertor in MXNet

[Caffe](http://caffe.berkeleyvision.org/) is a well-known and widely used deep learning framework. MXNet supports calling most Caffe operators (layers) and loss functions directly in its symbolic graph. Using your own customized Caffe layer is also effortless.

MXNet also has embedded [Torch modules and its tensor mathematical functions](https://github.com/dmlc/mxnet/blob/master/docs/how_to/torch.md).

This topic explains how to:

* Install MXNet with Caffe support

* Embed Caffe operators into MXNet's symbolic graph

## Install Caffe With MXNet


1. Download the official Caffe repository, [BVLC/Caffe](https://github.com/BVLC/caffe).
2. Download the [Caffe patch for the MXNet interface](https://github.com/BVLC/caffe/pull/4527.patch). Move the patch file under your Caffe root folder, and apply the patch by using `git apply patch_file_name`.
3. Install Caffe using the [official guide](http://caffe.berkeleyvision.org/installation.html).

## Compile with Caffe


1. If you haven't already, copy `make/config.mk` (for Linux) or `make/osx.mk` (for Mac) into the MXNet root folder as `config.mk`.
2. In the mxnet folder, open `config.mk` and uncomment the lines `CAFFE_PATH = $(HOME)/caffe` and `MXNET_PLUGINS += plugin/caffe/caffe.mk`. Modify `CAFFE_PATH` to your Caffe installation, if necessary.
3. To build with Caffe support, run `make clean && make`.

## Using the Caffe Operator (Layer)
Caffe's neural network operator and loss functions are supported by MXNet through `mxnet.symbol.CaffeOp` and `mxnet.symbol.CaffeLoss`, respectively.
For example, the following code shows a [multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP) network for classifying MNIST digits: [full code](https://github.com/dmlc/mxnet/blob/master/example/caffe/caffe_net.py):

### Python

```Python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")
act1 = mx.symbol.CaffeOp(data_0=fc1, prototxt="layer{type:\"TanH\"}")
fc2  = mx.symbol.CaffeOp(data_0=act1, num_weight=2, name='fc2', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 64} }")
act2 = mx.symbol.CaffeOp(data_0=fc2, prototxt="layer{type:\"TanH\"}")
fc3 = mx.symbol.CaffeOp(data_0=act2, num_weight=2, name='fc3', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 10}}")
mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
```

Let's break it down. First, `data = mx.symbol.Variable('data')` defines a variable as a placeholder for input.
Then, it's fed through Caffe operators with `fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")`.

The inputs to Caffe operators are named as data_i for i=0.  num_data-1 as `num_data` is the number of inputs. You can skip the argument, as the example does, if its value is 1. `num_weight` is the number of `blobs_`(weights). Its default value is 0 because many operators maintain no weight. `prototxt` is the configuration string.

To use the loss function in Caffe, replace the last line with:

```Python
label = mx.symbol.Variable('softmax_label')
mlp = mx.symbol.CaffeLoss(data=fc3, label=label, grad_scale=1, name='softmax', prototxt="layer{type:\"SoftmaxWithLoss\"}")
```
