<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

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
