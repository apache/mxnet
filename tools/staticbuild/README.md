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

# MXNet Static Build

This folder contains the core script used to build the static library. This README provides information on how to use the scripts in this folder. Please be aware, all of the scripts are designed to be run under the root folder.

## `build.sh`
This script is a wrapper around `build_lib.sh. It simplifies the build by
automatically identifing the system version, number of cores, and all
environment variable settings. Here are examples you can run with this script:

```
tools/staticbuild/build.sh cu102
```
This would build the mxnet package based on CUDA 10.2. Currently, we support variants cpu, native, cu90, cu92, cu100, and cu101. All of these variants expect native have MKL-DNN backend enabled. 

```
tools/staticbuild/build.sh cpu
```

This would build the mxnet package based on MKL-DNN.

To use CMake to build the `libmxnet.so` instead of the deprecated Makefile based
build logic, set the `CMAKE_STATICBUILD` environment variable. For example

```
CMAKE_STATICBUILD=1 tools/staticbuild/build.sh cpu
```

For the CMake build, you need to install `patchelf` first, for example via `apt
install patchelf` on Ubuntu systems.

As the result, users would have a complete static dependencies in `/staticdeps` in the root folder as well as a static-linked `libmxnet.so` file lives in `lib`. You can build your language binding by using the `libmxnet.so`.

## `build_lib.sh`
This script clones the most up-to-date master and builds the MXNet backend with a static library. In order to run the static library, you must set the following environment variables:

- `DEPS_PATH` Path to your static dependencies
- `PLATFORM` linux, darwin
- `VARIANT` cpu, cu*

It is not recommended to run this file alone since there are a bunch of variables need to be set.

After running this script, you would have everything you need ready in the `/lib` folder.

## `build_wheel.sh`
This script builds the python package. It also runs a sanity test.
