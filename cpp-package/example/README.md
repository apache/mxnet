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

# MXNet C++ Package Examples

## Building C++ examples

The examples in this folder demonstrate the **training** workflow. The **inference workflow** related examples can be found in [inference](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference>) folder.
Please build the MXNet C++ Package as explained in the [README](<https://github.com/apache/incubator-mxnet/tree/master/cpp-package#building-c-package>) File before building these examples manually.
The examples in this folder are built while building the MXNet library and cpp-package from source. However, they can be built manually as follows

From cpp-package/examples directory

-  Build all examples in release mode: **make all**
-  Build all examples in debug mode: **make debug**

By default, the examples are built to be run on GPU. To build examples to run on CPU:

-  Release: **make all MXNET\_USE\_CPU=1**
-  Debug: **make debug MXNET\_USE\_CPU=1**

The examples that are built to be run on GPU may not work on the non-GPU machines.
The makefile will also download the necessary data files and store in a data folder. (The download will take couple of minutes, but will be done only once on a fresh installation.)
