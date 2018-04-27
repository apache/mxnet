# MXNet C++ Package

To build the C++ package, please refer to [this guide](<https://mxnet.incubator.apache.org/install/build_from_source#build-the-c-package>).

A basic tutorial can be found at <https://mxnet.incubator.apache.org/tutorials/c++/basics.html>.

The example directory contains examples for you to get started.

## Building C++ examples in examples folder

From cpp-package/examples directory
-  Build all examples in release mode: **make all**
-  Build all examples in debug mode : **make debug**

By default, the examples are build to be run on CPU.
To build examples to run on GPU:
- Release: **make all MXNET_USE_CUDA=1**  
- Debug: **make debug MXNET_USE_CUDA=1**  


The makefile will also download the necessary data files and store in data folder. (The download will take couple of minutes, but will be done only once on a fresh installation.)
