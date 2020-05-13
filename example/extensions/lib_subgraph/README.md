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

Adding custom model partitioners in MXNet used to require deep understanding of the MXNet backend, including operator registration and other internal classes, followed by recompiling MXNet from source. This feature allows adding custom partitioners by dynamically loading external libraries at runtime.

This custom partitioner feature enables users to write custom model partitioning strategies without compiling against all of MXNet header files and dependencies. When a library containing custom partitioners is loaded dynamically, the components found in the library will be registered in MXNet so that users can use those natively just like other built-in components.

## Getting Started

### Have MXNet Ready

To run the following example, the build type of MXNet doesn’t matter since the custom partitioner doesn’t interact with the execution of other native MXNet features. Note that if you want to use your custom partitioners with models running on GPU, you still need an MXNet CUDA build. 

### Run An Example

You can start getting familiar with custom partitioners by running an example provided in the **example/extensions/lib_subgraph** directory. This example partitions `exp` and `log` operators into subgraphs. Go to the **lib_subgraph** directory and follow these steps:

1. Run `make`. The Makefile will generate the dynamic library **libsubgraph_lib.so** which is compiled from the `subgraph_lib.cc` file. This is the library you are going to load that contains everything for the custom partitioner.
2. Run `python test_subgraph.py`. It’ll first load the above library, find the components, register them in the MXNet backend, then partition the model and execute the operators like a regular MXNet operator and output the result. Below is the output when running the `python test_subgraph.py` command. Notice that it loads 2 operators: my_gemm and state_gemm.

```
[10:38:03] src/c_api/c_api.cc:286: Found 1 operators in library
[10:38:03] src/c_api/c_api.cc:350:       Op[0] _custom_subgraph_op
[10:38:03] src/c_api/c_api.cc:785: Found 1 partitioners in library
[10:38:03] src/c_api/c_api.cc:801:       Partitioner[0] myProp
[10:38:03] src/c_api/c_api.cc:821:             Strategy[0] strategy1 subgraphOp: '_custom_subgraph_op'
```

### Basic Files For Custom Partitioner Library

* **lib_subgraph/subgraph_lib.cc**: This file has a source code implementation of all required components to make a custom partitioner, it also shows registration of them so that they can be loaded by MXNet.

* **lib_subgraph/Makefile**: This file compiles the source code to a dynamic shared library, with a header file `include/mxnet/lib_api.h` from MXNet source code. Currently the custom operator is compatible with C++11 onwards.

* **lib_subgraph/test_subgraph.py**: This file calls `mx.library.load(‘libsubgraph_lib.so’)` to load the library containing the custom components, partitions the model using the `optimize_for` API, and prints outputs of the forward passes. The outputs should be the same as the regular MXNet forward pass without partitioning.

* **include/mxnet/lib_api.h**: This file from MXNet source code is the single header file needed to include all necessary data types and function prototypes for writing a custom operator library. You can either specify the include path in the `Makefile`, or copy the header file over to `example/extensions/lib_subgraph` folder. Note that apart from this header, the custom operator library is independent of MXNet source.

## Writing Custom Partitioner Library

To build your own library containing a custom partitioner, compose a C++ source file like `mypart_lib.cc`, include `lib_api.h` header file, and write your custom partitioner with these essential functions:
- `initialize` - Library Initialization Function
- `REGISTER_PARTITIONER ` - Partitioner Registration Macro
- `mySupportedOps ` - Operator Support

Then compile it to the `mypart_lib.so` dynamic library using the following command:

```bash
g++ -shared -fPIC -std=c++11 mypart_lib.cc -o libmypart_lib.so -I ../../../include/mxnet
```

Finally, you can write a Python script to load the library and partition a model with your custom partitioner:

```python
import mxnet as mx
mx.library.load(‘libmyop_lib.so’)
sym, _, _ = mx.model.load_checkpoint('mymodel', 0) 

# Symbol/Module flow
sym2 = sym.optimize_for("myPart")

# Gluon flow 1
sym_block = nn.SymbolBlock(sym, inputs)
sym_block.hybridize(backend='myPart')

# Gluon flow 2
sym_block = nn.SymbolBlock(sym, inputs)
sym_block.optimize_for(x, backend='myPart')
```

In the Gluon hybridize flow, the model is actually hybridized during the first inference, rather than immediately when calling `hybridize`. This hybridize-based flow is useful if a user expects to run inference immediately after hybridizing. But for users than just want to partition but not run a whole forward pass, the `optimize_for` API combines the hybrdize/forward APIs but does not run a forward pass. After calling `optimize_for` users can `export` their model immediately without running a forward pass. 

### Using a Custom Partitioner Library

Partitioning APIs in MXNet are available in both Symbol and Gluon APIs. For the Symbol API, the `optimize_for` API can be called on Symbol objects to return a partitioned Symbol.

```
optimize_for(backend, args=None, aux=None, ctx=None, **kwargs)
```

The `optimize_for` API takes at least 1 argument, `backend` which is a string that identifies which backend to partition the model for. The `args` and `aux` arguments are optional and take a list of NDArray or dict of str to NDArray. They are used to infer shapes and types and before partitioning, and passed to the backend to use during compilation. The `ctx` argument is optional and takes a device context to infer storage types. It also takes any other user-specified options that will be passed to the backend partitioning APIs.

For the Gluon API, the `hybridize` API can be called on HybridBlocks to partition the internal CachedOp Symbol.

```
hybridize(backend=None, backend_opts=None, **kwargs)
```

The `hybridize` function prepares the HybridBlock to be converted into a backend symbol. The `backend` argument is a string that identifies which backend that will partition the model. The `backend_opts` takes other user-specified options that will be passed to the backend partitioning APIs. The actual partitioning takes place during the forward pass.

If you just want to partition the HybridBlock but not run a complete forward pass, you can use the `optimize_for` API that combines the work done in the `hybridize` API with part of the work done in the forward pass.

```
optimize_for(x, backend=None, backend_opts=None, **kwargs)
```

When the `optimize_for` API is called on a HybridBlock it partitions immediately. This lets users export the partitioned model without running a complete forward pass.

```
block.optimize_for(x, backend='myPart')
block.export('partitioned')
```

But you can also use `optimize_for` in place of `hybridize` and run inference immediately after too.

```
block.optimize_for(x, backend='myPart')
block(x)
```

### Writing A Custom Partitioner

There are several essential building blocks for making a custom partitioner:

* [initialize](./subgraph_lib.cc#L261):
    * This function is the library initialization function necessary for any dynamic libraries. It lets you check if the user is using a compatible version of MXNet. Note that this `version` parameter is passed from MXNet when library is loaded.

            MXReturnValue initialize(int version)

* [supportedOps](./subgraph_lib.cc#L179):
    * This function provides a copy of the model graph as a JSON string, and provides an interface for identifying which operators should be partitioned into a subgraph. Also this is where a custom partitioner can validate the options specified by the user.

            MXReturnValue supportedOps(
                std::string json,
                std::vector<bool>& ids,
                std::unordered_map<std::string, std::string>& options)

* [REGISTER_PARTITIONER(my_part_name)](./subgraph_lib.cc#L257):
    * This macro registers the custom partitioner and its properties to MXNet by its name. Notice that a partitioner can have multiple partitioning strategies. This enables multiple *passes* to be run in a single partitioning call from the user. The first argument to `addStrategy` is a user-specified name. The second argument is the `supportedOps` function. The third argument is the name of the subgraph operator to create for each subgraph created during partitioning (see below for more info about subgraph operators). The `setReviewSubgraph` API registers a callback function that is called for each subgraph created during partitioning (more on this below). Notice that the first argument to this function is the strategy to associate with and the second argument is the `reviewSubgraph` function.

            REGISTER_PARTITIONER(my_part_name)
            .addStrategy("strategy1", supportedOps, "_custom_subgraph_op")
            .setReviewSubgraph("strategy1", reviewSubgraph);


Also there are some optional functions you can specify:

* [reviewSubgraph](./subgraph_lib.cc#L219):
    * This function provides an opportunity to accept/reject a subgraph after MXNet partitions it. It also allows specifying custom attributes on the subgraph (ie. user-generated IDs). If you do not register this function, subgraphs will be accepted by default. 

            MXReturnValue reviewSubgraph(
                std::string json,
                int subgraph_id,
                bool* accept,
                std::unordered_map<std::string, std::string>& options,
                std::unordered_map<std::string, std::string>& attrs,
                std::map<std::string, MXTensor>& args,
                std::map<std::string, MXTensor>& aux)

Let’s take a closer look at those registry functions:

* **supportedOps**: This function takes four arguments. The 1st argument is a JSON string of the model architecture graph, where nodes are inputs/params/weights and edges are data dependencies. The graph is pre-sorted in topological order. The 2nd argument is an array of booleans, one for each operator in the model. When traversing the graph, operators to be partitioned into subgraphs are identified and an entry is set to `true` for the index in the `ids` array corresponding to the node ID. The last argument is the map of options specified by the user. Users can pass custom options to the partitioner and they are passed to this function in the `options` map. 

* **reviewSubgraph**: This function takes five arguments. The 1st argument is a JSON string of the newly partitioned subgraph. The 2nd argument is the subgraph ID, this is just a number MXNet uses to identify this particular subgraph (it starts at zero and increments, unique for each subgraph in the model). The 3rd argument is an output to be set in this function to tell MXNet whether to accept (value: `true`) or reject (value: `false`) the subgraph. You might want to reject a subgraph if it doesnt include all the operators you want, for example. The `options` map is the same one passed to the `supportedOps` API. The 4th argument is the map of options specified by the user. The 5th argument is a map of attributes that should be set on the created subgraph. These attributes will be available later at runtime, and provides a mechanisn to pass info from partition-time to runtime. The last argument is the map of params/weights/args to the model and the associated names. For inputs the the subgraph that come directly from the params/weights of the model, you can look up the name of the input in this map to get the actual tensor values.

### Writing A Custom Subgraph Operator

A partitioning strategy specifies how to partition a model and isolate operators into subgraphs. In MXNet, subgraphs are just a [stateful operator](../lib_custom_op#writing-stateful-custom-operator). Subgraph operators have an extra attribute called `MX_STR_SUBGRAPH_SYM_JSON` that maps to a JSON string of the subgraph. The expectation is that when a subgraph operator executes a forward/backward call, it executes all of the operators in the subgraph. 

When registering a custom subgraph operator, all thats needed is to register a `createOpState` function and to set that the operator is a subgraph operator by calling the `setIsSubgraphOp` API like:

```
REGISTER_OP(my_subgraph_op)
.setIsSubgraphOp()
.setCreateOpState(createOpState, "cpu");
```

### Parsing a JSON string

To simplify custom partitioner libraries, basic JSON parsing utility functions have been implemented in the `lib_api.h` header file. You create a `JsonParser` object and parse the string by calling the `parse_to_json` API like:

```c++
JsonParser parser;
JsonVal json_val = parser.parse_to_json(json_string);
```

A `JsonVal` is a class that represents the nodes in a JSON structure. You can check the type of a node (num, str, list, or map) by comparing the `JsonVal.type` to `STR`, `NUM`, `LIST`, or `MAP`. Then you can get that value from the node like:

```c++
switch(json_val.type) {
  case STR:
    std::string str = json_val.str;
    break;
  case NUM:
    int num = json_val.num;
    break;
  case LIST:
    std::vector<JsonVal> list = json_val.list;
    break;
  case MAP:
    std::map<JsonVal, JsonVal> map = json_val.map;
    break;
  default:
    // error
}
```

There are also convenience constructors for creating `JsonVal` objects for strings and numbers like `JsonVal("myKey")` or `JsonVal(42)`. This makes it easy to get specific keys from a map like `json_val.map[JsonVal("nodes")]`.