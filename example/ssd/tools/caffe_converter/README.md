# Convert Caffe Model to Mxnet Format

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
To convert ssd caffemodels, Use: `python convert_model.py prototxt caffemodel outputprefix`

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
