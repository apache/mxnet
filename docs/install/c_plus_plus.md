## Build the C++ package
The C++ package has the same prerequisites as the MXNet library.

To enable C++ package, just add `USE_CPP_PACKAGE=1` in the [build from source](build_from_source.html) options when building the MXNet shared library.

For example to build MXNet with GPU support and the C++ package, OpenCV, and OpenBLAS, from the project root you would run:

```bash
cmake -DUSE_CUDA=1 -DUSE_CUDA_PATH=/usr/local/cuda -DUSE_CUDNN=1 -DUSE_MKLDNN=1 -DUSE_CPP_PACKAGE=1 -GNinja .
ninja -v
```

You may also want to add the MXNet shared library to your `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=~/incubator-mxnet/lib
```

Setting the `LD_LIBRARY_PATH` is required to run the examples mentioned in the following section.

## C++ Example Code
You can find C++ code examples in the `cpp-package/example` folder of the MXNet project. Refer to the [cpp-package's README](https://github.com/apache/incubator-mxnet/tree/master/cpp-package) for instructions on building the examples.

## Tutorials

* [MXNet C++ API Basics](https://mxnet.incubator.apache.org/tutorials/c++/basics.html)

## Related Topics

* [Image Classification using MXNet's C Predict API](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp)
