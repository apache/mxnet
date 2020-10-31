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

External Operators Example and Tutorial
=======================================

## Introduction

Extending MXNet with custom components used to mean distributing a custom fork. This feature allows adding custom components to MXNet by dynamically loading external libraries at runtime. Currently it is only supported on Linux systems (Windows and Mac are __NOT__ supported). 

## Getting Started

### Have MXNet Ready

For this tutorial, clone MXNet from source like:
```
git clone https://github.com/apache/incubator-mxnet.git --recursive --init
```

Build MXNet like:
```
cp config/linux.cmake config.cmake
mkdir build
cd build
cmake ..
cmake --build .
```

## Run An Example

This example shows compiling a custom backend operator and then dynamically loading it into MXNet at runtime. Go to the **lib_external_ops** directory and follow these steps:

1. Touch or modify the **min_ex.cc** and/or **min_ex-inl.h** file(s)
2. Go into the **build** directory that was created when building MXNet.
3. Run `cmake .. -DBUILD_EXTENSION_PATH=$(pwd)/../example/extensions/lib_external_ops`
4. Run `cmake --build .`
5. Go to the **example/extensions/lib_external_ops** directory again
6. Run `python test_loading.py` to execute the test program. You should see the following output:
```
Operator not registered yet
MXNet version 20000 supported
[]
Operator executed successfully
```

## Writing an External Operator Library
To build your own library containing custom components, compose a C++ source file like `mycomp_lib.cc`, include the `lib_api.h` header file, compile the `lib_api.cc` file, and implement the following required function:
- `initialize` - Library Initialization Function

Then create a CMakeLists.txt file and set `mxnet` as a link library like:
```
add_library(external_lib SHARED ${SRCS})
target_link_libraries(external_lib PUBLIC mxnet)
```

Next, build MXNet and set the path to your directory with the CMakeLists.txt file via the `BUILD_EXTENSION_PATH` option. This will build your library with all of the MXNet includes. 
