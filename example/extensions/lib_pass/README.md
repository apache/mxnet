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

Custom Graph Pass Example and Tutorial
=======================================

## Introduction

Adding custom graph passes in MXNet used to require deep understanding of the MXNet backend, including nnvm pass registration and other internal classes, followed by recompiling MXNet from source. This feature allows adding custom graph passes by dynamically loading external libraries at runtime.

This custom graph pass feature enables users to write custom model modification strategies without compiling against all of MXNet header files and dependencies. When a library containing custom passes is loaded dynamically, the components found in the library will be registered in MXNet so that users can use those natively just like other built-in components.

## Getting Started

### Have MXNet Ready

To run the following example, the build type of MXNet doesn’t matter since the custom pass doesn’t interact with the execution of other native MXNet features. Note that if you want to use your custom pass with models running on GPU, you still need an MXNet CUDA build. 

### Run An Example

You can start getting familiar with custom passes by running an example provided in the **example/extensions/lib_pass** directory. The `myPass` example just copies the input graph to the output. Go to the **lib_pass** directory and follow these steps:

1. Run `make`. The Makefile will generate the dynamic library **libpass_lib.so** which is compiled from the `pass_lib.cc` file. This is the library you are going to load that contains everything for the custom pass.
2. Run `python test_pass.py`. It’ll first load the above library, find the components, register them in the MXNet backend, then execute the pass on the model and execute the operators like a regular MXNet operator and output the result. Below is the output when running the `python test_pass.py` command. Notice that it loads 2 passes: myPass and jsonPass.

```
[10:38:03] src/c_api/c_api.cc:286: Found 0 operators in library
[10:38:03] src/c_api/c_api.cc:785: Found 0 partitioners in library
[07:14:00] src/c_api/c_api.cc:887: Found 2 graph passes in library
[07:14:00] src/c_api/c_api.cc:902:       Graph Pass [0] myPass
[07:14:00] src/c_api/c_api.cc:902:       Graph Pass [1] jsonPass
```

### Basic Files For Custom Pass Library
* **lib_pass/pass_lib.cc**: This file has a source code implementation of all required components to make a custom pass, it also shows registration of them so that they can be loaded by MXNet.
* **lib_pass/Makefile**: This file compiles the source code to a dynamic shared library, with a header file `include/mxnet/lib_api.h` from MXNet source code. Currently the custom pass is compatible with C++11 onwards.
* **lib_pass/test_pass.py**: This file calls `mx.library.load(‘libpass_lib.so’)` to load the library containing the custom components, executes the pass on the model using the `optimize_for` API, and prints outputs of the forward passes. The outputs should be the same as the regular MXNet forward pass without running the pass.
* **include/mxnet/lib_api.h**: This file from MXNet source code is the single header file needed to include all necessary data types and function prototypes for writing a custom library. You can either specify the include path in the `Makefile`, or copy the header file over to `example/extensions/lib_pass` folder. Note that apart from this header, the custom library is independent of MXNet source.
## Writing Custom Pass Library
To build your own library containing a custom pass, compose a C++ source file like `mypass_lib.cc`, include `lib_api.h` header file, and write your custom pass with these essential functions:
- `initialize` - Library Initialization Function
- `REGISTER_PASS` - Pass Registration Macro
- `graphPass` - Pass Implementation
Then compile it to the `mypass_lib.so` dynamic library using the following command:
```bash
g++ -shared -fPIC -std=c++11 mypass_lib.cc -o libmypass_lib.so -I ../../../include/mxnet
```

Finally, you can write a Python script to load the library and execute your pass on a model:

```python
import mxnet as mx
mx.library.load(‘libmypass_lib.so’)
sym, _, _ = mx.model.load_checkpoint('mymodel', 0) 
# Symbol/Module flow
sym2 = sym.optimize_for("myPass")
# Gluon flow 1
sym_block = nn.SymbolBlock(sym, inputs)
sym_block.hybridize(backend='myPass')
# Gluon flow 2
sym_block = nn.SymbolBlock(sym, inputs)
sym_block.optimize_for(x, backend='myPass')
```

### Using a Custom Pass Library

APIs in MXNet are available in both Symbol and Gluon APIs. For the Symbol API, the `optimize_for` API can be called on Symbol objects to return a new Symbol post graph pass.

```
optimize_for(backend, args=None, aux=None, ctx=None, **kwargs)
```

The `optimize_for` API takes at least 1 argument, `backend` which is a string that identifies which backend to use to optimize the model. The `args` and `aux` arguments are optional and take a list of NDArray or dict of str to NDArray. They are used to infer shapes and types and before executing the graph pass. The `ctx` argument is optional and takes a device context to infer storage types. It also takes any other user-specified options that will be passed to the backend APIs.

For the Gluon API, the `hybridize` API can be called on HybridBlocks to execute a graph pass on the internal CachedOp Symbol.

```
hybridize(backend=None, backend_opts=None, **kwargs)
```

The `hybridize` function prepares the HybridBlock to be converted into a backend symbol. The `backend` argument is a string that identifies which pass that will be executed on the model. The `backend_opts` takes other user-specified options that will be passed to the backend APIs. The actual pass runs once just before the first the forward pass.

If you just want to run a graph pass on the HybridBlock but not run a complete forward pass, you can use the `optimize_for` API that combines the work done in the `hybridize` API with part of the work done in the forward pass.

```
optimize_for(x, backend=None, backend_opts=None, **kwargs)
```

When the `optimize_for` API is called on a HybridBlock it runs the graph pass immediately. This lets users export the modified model without running a complete forward pass.

```
block.optimize_for(x, backend='myPass')
block.export('optimized')
```

But you can also use `optimize_for` in place of `hybridize` and run inference immediately after too.

```
block.optimize_for(x, backend='myPass')
block(x)
```

### Writing A Custom Graph Pass

There are several essential building blocks for making a custom pass:

* [initialize](./pass_lib.cc#44):
    * This function is the library initialization function necessary for any dynamic libraries. It lets you check if the user is using a compatible version of MXNet. Note that this `version` parameter is passed from MXNet when library is loaded.

            MXReturnValue initialize(int version)

* [graphPass](./pass_lib.cc#31):
    * This function provides a copy of the model graph as a JSON string, and provides an interface for returning a modified model JSON string. Also this is where a custom pass can validate the options specified by the user.

            MXReturnValue graphPass(
                const std::string& in_graph,
                const std::string** out_graph,
                const std::unordered_map<std::string, std::string>& options,
                const std::unordered_map<std::string, MXTensor>& args,
                const std::unordered_map<std::string, MXTensor>& aux,
                const PassResource& res)

* [REGISTER_PASS(my_pass_name)](./pass_lib.cc#L41):
    * This macro registers the custom pass and its properties to MXNet by its name. The argument to `setBody` is the `graphPass` function.

            REGISTER_PASS(my_pass_name)
            .setBody(graphPass);

Let’s take a closer look at those registry functions:

* **graphPass**: This function takes six arguments. The 1st argument is a JSON string of the model architecture graph, where nodes are inputs/params/weights and edges are data dependencies. The graph is pre-sorted in topological order. The 2nd argument is a pointer to a pointer of a JSON model string. It is expected users will dereference and assign the address of their output string allocated with `new` and `delete` will be called on it automatically. The third argument is the map of options specified by the user. Users can pass custom options to the pass and they are passed to this function in the `options` map. The fourth and fifth arguments are the named tensor mapping for the args and aux params for the model. They will contain the model params if the user provides them to the `optimize_for` API. The last argument is the `PassResource` object for memory allocation and other utilities. The details of `PassResource` are covered in the section below

### Pass Resource

Some graph passes require allocating new NDArrays to add/replace model params. The `alloc_arg` and `alloc_aux` APIs enabling allocating new NDArrays and integrating them with the user-provide args and aux params. Both APIs have the following signature:

```
    MXTensor* alloc_xxx(const std::string& name,
                        const std::vector<int64_t>& shapes,
                        const MXContext &ctx,
                        MXDType dtype)
```

If the `name` provided matches the name of an existing param it replaces the previous one. Otherwise it adds a new param to the appropriate arg/aux set.

### Parsing a JSON string

To simplify custom libraries, basic JSON parsing utility functions have been implemented in the `lib_api.h` header file. You create a `JsonParser` object and parse the string by calling the `parse_to_json` API like:

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
