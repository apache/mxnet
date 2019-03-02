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

# Overview

This folder contains scripts for building the dependencies from source. The static libraries from
the build artifacts can be used to create self-contained shared object for mxnet through static
linking.

# Settings

The scripts use the following environment variables for setting behavior:

`DEPS_PATH`: the location in which the libraries are downloaded, built, and installed.
`PLATFORM`: name of the OS in lower case. Supported options are 'linux' and 'darwin'.

It also expects the following build tools in path: make, cmake, tar, unzip, autoconf, nasm

# FAQ

## Build failure regarding to gcc, g++, gfortran
Currently, we only support gcc-4.8 build. It's your own choice to use a higher version of gcc. Please make sure your gcc, g++ and gfortran always have the same version in order to eliminate build failure.

## idn2 not found
This issue appeared in the OSX build with XCode version 8.0 above (reproduced on 9.2). Please add the following build flag in `curl.sh` if your XCode version is more than 8.0:
```
--without-libidn2
``` 