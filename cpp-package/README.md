<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

# MXNet - C++ API

The MXNet C++ Package provides C++ API bindings to the users of MXNet.  Currently, these bindings are not available as standalone package.
The users of these bindings are required to build this package as mentioned below.

## Building C++ Package

The cpp-package directory contains the implementation of C++ API. Users are required to build this directory or package before using it. 
**The cpp-package is built while building the MXNet shared library, *libmxnet.so*, with *USE\_CPP\_PACKAGE* option turned on. Please follow the steps to build the C++ package**

### Steps to build the C++ package:
1.  Building the MXNet C++ package requires building MXNet from source.
2.  Clone the MXNet GitHub repository **recursively** to ensure the code in submodules is available for building MXNet.
	```
	git clone --recursive https://github.com/apache/incubator-mxnet mxnet
	```

3.  Install the [recommended dependencies](https://mxnet.apache.org/versions/master/get_started/build_from_source.html#installing-mxnet's-recommended-dependencies) and [optional dependencies](https://mxnet.apache.org/versions/master/get_started/build_from_source.html#overview-of-optional-dependencies-and-optional-features) for building MXNet from source.
4.  There is a configuration file for cmake, [config/*.cmake](<https://github.com/apache/incubator-mxnet/tree/master/config>) that contains all the compilation options. You can edit this file and set the appropriate options prior to running the **cmake** command.
5.  Please refer to  [cmake configuration files](https://github.com/apache/incubator-mxnet/blob/970a2cfbe77d09ee610fdd70afca1a93247cf4fb/config/linux_gpu.cmake#L18-L37) for more details on how to configure and compile MXNet.
6.  For enabling the build of C++ Package, set the **-DUSE\_CPP\_PACKAGE = 1** in cmake options.

### Cross-Compilation steps:
1.  Build the C++ package for the **host** platform to generate op.h file.
2.  Remove the following line in [CMakeLists.txt](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/CMakeLists.txt#L15>).
    ```
	COMMAND python OpWrapperGenerator.py $<TARGET_FILE:mxnet>
	``` 
3.  Re-configure cmake for cross-compilation to build the **target** C++ package.

## Usage

In order to consume the C++ API please follow the steps below.

1. Ensure that the MXNet shared library is built from source with the **USE\_CPP\_PACKAGE = 1**.
2. Include the [MxNetCpp.h](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/include/mxnet-cpp/MxNetCpp.h>) in the program that is going to consume MXNet C++ API.
	```c++
	#include <mxnet-cpp/MxNetCpp.h>
	```
3. While building the program, ensure that the correct paths to the directories containing header files and MXNet shared library.
4. The program links the MXNet shared library dynamically. Hence the library needs to be accessible to the program during runtime. This can be achieved by including the path to the shared library in the environment variable  **LD\_LIBRARY\_PATH** for Linux, Mac. and Ubuntu OS and **PATH** for Windows OS.


## Tutorial

A basic tutorial can be found at <https://mxnet.apache.org/api/cpp/docs/tutorials/basics>.

## Examples

The example directory contains examples for you to get started. Please build the MXNet C++ Package before building the examples.
