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

# nGraph - MXNet Integration
MXNet nGraph integration is based on [Unified integration with external backend libraries](https://cwiki.apache.org/confluence/display/MXNET/Unified+integration+with+external+backend+libraries)

After building MXNet with nGraph support, users can enable nGraph backend by setting `MXNET_SUBGRAPH_BACKEND="ngraph"`environmental variable. 

Gluon support is experimental and may or may not yield good performance. Gluon-NGraph 
integration can be enabled by setting the environmental variable `MXNET_NGRAPH_GLUON=1`

## Building with nGraph support
MXNet's experimental support for the Intel nGraph graph compiler can be enabled
using MXNet's build system. Current support is for Linux-based OS's, Mac and Windows
support will be added in future releases.

When building MXNet with experimental nGraph integration enabled, MXNet's build
system builds its own copy of the nGraph-supplied libraries.  Upon successful
completion of an nGraph-enabled build, these libraries and related symbolic links
can be found in the same build directory as `libmxnet.so`.

If building with gnu make, use the command:

`make -j USE_NGRAPH=1`

If building with cmake, use the command:

`mkdir build && cd build && cmake ../ -DUSE_NGRAPH=1 && make -j`

## Runtime environment variables
Some environment variables influence the behavior of the
nGraph-enabled MXNet software and supporting libraries.  Here is a partial list of those variables:

| Variable  | Description |
| :-------- | :---------- |
| `OMP_NUM_THREADS`            | Suggested value: `16`.  For more information please see [here](https://software.intel.com/en-us/mkl-windows-developer-guide-setting-the-number-of-threads-using-an-openmp-environment-variable) |
| `KMP_AFFINITY`               | Suggested value: `granularity=fine,compact,1,0`.  For more information please see [here](https://software.intel.com/en-us/node/522691). |
| `MXNET_NGRAPH_VERBOSE_GRAPH` | When set to `1`, nGraph-enabled MXNet will create in the current directory a JSON file representing each subgraph being compiled by the nGraph library.  Each of these JSON files is a graph serialization that can be loaded by nGraph's `ngraph::deserialize`  functions. |

## Supported nGraph back-ends
The nGraph library supports a number of hardware and software backends, including `"CPU"`, `"INTERPETER"` (reference kernels), `"GPU"`, and `"IntelGPU"`. Current experimental integration enables `"CPU"` backend by default. More backends will be supported in future releases.
