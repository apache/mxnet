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

Custom Partitioner Example and Tutorial
=======================================

## Introduction

Adding custom model partitioners in MXNet used to require understanding of MXNet backend operator registration and recompiling of MXNet with all its dependencies. So our approach for adding custom partitioners is to enable dynamic loading of C++ partitioning code compiled in external libraries at runtime.

Custom partitioners enable users to write custom model partitioning strategies without compiling against all of MXNet header files and dependencies. When a library containing custom partitioners is loaded dynamically, the components found in the library will be re-registered in MXNet so that users can use those natively just like other built-in components.

## Getting Started

### Have MXNet Ready

First you should install MXNet either from compiling from source code or download from nightly build. It doesn’t matter if the build comes with CUDA or MKLDNN. The custom partitioning APIs do not interact with the execution of other native MXNet operators.

### Run An Example:

You can start getting familiar with custom partitioners by running an example provided in the **example/extensions/lib_subgraph** directory. This example partitions `exp` and `log` operators into subgraphs. Go to `lib_subgraph` directory and follow these steps:

1. Run `make`. The Makefile will generate a dynamic library ** libsubgraph_lib.so** compiled from `subgraph_lib.cc`. This is the library you are going to load that contains everything for the custom partitioner.
2. Run `python test_subgraph.py`. It’ll first load the above .so library, find the components, register them in the MXNet backend, print "Found x", then partition the model and execute the operators like a regular MXNet operator and output the result.

### Basic Files For Custom Partitioner Library:

* ** lib_subgraph/subgraph_lib.cc**: This file has a source code implementation of all required components of a custom partitioner, as well as the registration of the custom components.

* ** lib_subgraph/Makefile**: Compile source code to a dynamic shared library, with a header file `include/mxnet/lib_api.h` from MXNet source code. Currently the custom operator is compatible with C++11 onwards.

* ** lib_subgraph/test_subgraph.py**: This file calls `mx.library.load(‘libsubgraph_lib.so’)` to load the library containing the custom components, partitions the model using the `optimize_for` API, and prints outputs of the forward passes. The outputs should be the same as the regular MXNet forward pass without partitioning.

## Writing Custom Partitioner Library:

For building a library containing your own custom partitioner, compose a C++ source file like `mypart_lib.cc`, include `lib_api.h` header file, and write your custom operator implementation with these essential functions:
- `initialize` - Library Initialization Function
- `REGISTER_PARTITIONER ` - Partitioner Registration Macro
- `mySupportedOps ` - Operator Support

Then compile it to `mypart_lib.so` dynamic library using the following command:
```bash
g++ -shared -fPIC -std=c++11 mypart_lib.cc -o libmypart_lib.so -I ../../../include/mxnet
```

Finally, you can write a Python script to load the library and partition a model with your custom partitioner:
```python
import mxnet as mx
mx.library.load(‘libmyop_lib.so’)
sym, _, _ = mx.model.load_checkpoint('mymodel', 0) 
sym.optimize_for("myPart")
```

### Writing A Custom Partitioner:

There are several essential building blocks for making a custom partitioner:

* [initialize](./subgraph_lib.cc#L242):
    * This function is the library initialization function necessary for any dynamic libraries. It lets you check if the user is using a compatible version of MXNet. Note that this `version` parameter is passed from MXNet when library is loaded.

            MXReturnValue initialize(int version)

* [supportedOps](./subgraph_lib.cc#L179):
    * This function provides a copy of the model graph as a JSON string, and provides an interface for identifying which operators should be partitioned into a subgraph. Also this is where a custom partitioner can validate the options specified by the user.

            MXReturnValue supportedOps(
                std::string json,
                const int num_ids,
                int *ids,
                std::unordered_map<std::string, 
                                   std::string>& options)

* [REGISTER_PARTITIONER(my_part_name)](./subgraph_lib.cc#L238):
    * This macro registers the custom partitioner and its properties to MXNet by its name. Notice that a partitioner can have multiple partitioning strategies. This enables multiple *passes* to be run in a single partitioning call from the user. The first argument to `addStrategy` is a user-specified name. The second argument is the `supportedOps` function. The third argument is the name of the subgraph operator to create for each subgraph created during partitioning (see below for more info about subgraph operators). The `setAcceptSubgraph` API registers a callback function that is called for each subgraph created during partitioning (more on this below). Notice that the first argument to this function is the strategy to associate with and the second argument is the `acceptSubgraph`.

            REGISTER_PARTITIONER(my_part_name)
            .addStrategy("strategy1", 
                          supportedOps, 
                          "_custom_subgraph_op")
            .setAcceptSubgraph("strategy1", 
                                acceptSubgraph);


Also there are some optional functions you can specify:

* [acceptSubgraph](./subgraph_lib.cc#L220):
    * This function provides an opportunity to accept/reject a subgraph after MXNet partitions it. It also allows specifying custom attributes on the subgraph (ie. user-generated IDs). If you do not register this function, subgraphs will be accepted by default. Only implement this function to set attributes or conditionally reject subgraphs. 

            MXReturnValue acceptSubgraph(
                std::string json,
                int subraph_id,
                bool* accept,
                std::unordered_map<std::string, 
                                   std::string>& options,
                std::unordered_map<std::string, 
                                   std::string>& attrs)

Let’s take a closer look at those registry functions:

* **supportedOps**: This function takes four arguments. The 1st argument is a JSON string of the model architecture graph, where nodes are inputs/params/weights and edges are data dependencies. The graph is pre-sorted in topological order. When traversing the graph, operators to be partitioned into subgraphs are identified and an entry is set to `1` for the node ID in the `ids` array. Users can pass custom options to the partitioner and they are passed to the function in the `options` map. 

* **acceptSubgraph**: This function takes five arguments. The 1st argument is a JSON string of the newly partitioned subgraph. It can be analyzed and accepted/rejected by setting `true`/`false` for the `accept` input. The `options` map is the same one passed to the `supportedOps` API. The `attrs` map provides an API to add user-specified attributes to the subgraph. These attributes will be available at runtime when the subgraph is executed and provides a way to pass info from partitioning-time to runtime. 

### Writing A Custom Subgraph Operator:

A partitioning strategy specifies how to partition a model and isolate operators into subgraphs. In MXNet, subgraphs are just a [stateful operator](../lib_custom_op#writing-stateful-custom-operator). Subgraph operators have an extra attribute called `SUBGRAPH_SYM_JSON` that maps to a JSON string of the subgraph. The expectation is that when a subgraph operator executes a forward/backward call, it executes all of the operators in the subgraph. 
