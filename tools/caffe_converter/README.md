# Convert Caffe Model to Mxnet Format

### Build

Either [Caffe's python package](http://caffe.berkeleyvision.org/installation.html) or [Google protobuf](https://developers.google.com/protocol-buffers/?hl=en) is required. The latter is often much easier to install:  

1. We first install the protobuf compiler. If you compiled mxnet with `USE_DIST_KVSTORE = 1` then it is already built. Otherwise, install `protobuf-compiler` by your favor package manager, e.g. `sudo apt-get install protobuf-compiler` for ubuntu and `sudo yum install protobuf-compiler` for redhat/fedora. 

2. Then install the protobuf's python binding. For example `sudo pip install protobuf`

Now we can build the tool by running `make` in the current directory.

### How to use

Use `./run.sh model_name` to download and convert a model. E.g. `./run.sh vgg19`

### Note

* We have verified the results of VGG_16/VGG_19 model and BVLC_googlenet results from Caffe model zoo.
* The tool only supports single input and single output network.
* The tool can only work with the L2LayerParameter in Caffe.
