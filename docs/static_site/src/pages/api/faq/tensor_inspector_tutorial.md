---
layout: page_category
title: Use TensorInspector to Help Debug Operators
category: faq
permalink: /api/faq/tensor_inspector_tutorial
---
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

# Use TensorInspector to Help Debug Operators

## Introduction

When developing new operators, developers need to deal with tensor objects extensively. This new utility, Tensor Inspector, mainly aims to help developers debug by providing unified interfaces to print, check, and dump the tensor value. To developers' convenience, this utility works for all the three data types: Tensors, TBlobs, and NDArrays. Also, it supports both CPU and GPU tensors.


## Usage 

This utility is located in `src/common/tensor_inspector.h`. To use it in any operator code, just include it using `#include "{path}/tensor_inspector.h"`, construct an `TensorInspector` object, and call the APIs on that object. You can run any script that uses the operator you just modified then.

The screenshot below shows a sample usage in `src/operator/nn/convolution-inl.h`.

![tensor_inspector_example_usage](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/faq/tensor_inspector_tutorial/tensor_inspector_example_usage.png)


## Functionalities/APIs

### Create a TensorInspector Object from Tensor, TBlob, and NDArray Objects

You can create a `TensorInspector` object by passing in two things: 1) an object of type `Tensor`, `Tbob`, or `NDArray`, and 2) an `RunContext` object.

Essentially, `TensorInspector` can be understood as a wrapper class around `TBlob`. Internally, the `Tensor`, `Tbob`, or `NDArray` object that you passed in will be converted to a `TBlob` object. The `RunContext` object is used when the tensor is a GPU tensor; in such a case, we need to use the context information to copy the data from GPU memory to CPU/main memory.

Following are the three constructors:

```c++
// Construct from Tensor object
template<typename Device, int dimension, typename DType MSHADOW_DEFAULT_DTYPE>
TensorInspector(const mshadow::Tensor<Device, dimension, DType>& ts, const RunContext& ctx);

// Construct from TBlob object
TensorInspector(const TBlob& tb, const RunContext& ctx);

// Construct from NDArray object
TensorInspector(const NDArray& arr, const RunContext& ctx):
```

### Print Tensor Value (Static) 

To print out the tensor value in a nicely structured way, you can use this API:

```c++
void print_string();
```

This API will print the entire tensor to `std::cout` and preserve the shape (it supports all dimensions from 1 and up). You can copy the output and interpret it with any `JSON` loader. You can find some useful information about the tensor on the last line of the output. Refer to the case below, we are able to know that this is a float-typed tensor with shape 20x1x5x5.

![tensor_inspector_to_string](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/faq/tensor_inspector_tutorial/tensor_inspector_to_string.png)

If instead of printing the tensor to `std::cout`, you just need a `string`, you can use this API:
```c++
std::string void to_string();
```

### Interactively Print Tensor Value (Dynamic) 

Sometimes at compilation time, you may not know which part of a tensor to inspect. Also, it may be nice to pause the operator control flow to “zoom into” a specific, erroneous part of a tensor multiple times until you are satisfied. In this regard, you can use this API to interactively inspect the tensor:

```c++
void  interactive_print(std::string tag =  "") {
```

This API will set a "break point" in your code. When that "break point" is reached, you will enter a loop that will keep asking you for further command input. In the API call, `tag` is an optional parameter to give the call a name, so that you can identify it when you have multiple `interactive_print()` calls in different parts of your code. A visit count will tell you how many times you stepped into this particular "break point", should this operator be called more than once. Note that all `interactive_print()` calls are properly locked, so you can use it in many different places without issues.

![tensor_inspector_interactive_print](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/faq/tensor_inspector_tutorial/tensor_inspector_interactive_print.png)

There are many useful commands available, as described in the previous screenshot: you can type "e" to print out the entire tensor, "d" to dump the tensor to file (see below), "b" to break from this command loop, and "s" to skip all future `interactive_print()`. Most importantly, in this screen, you can specify a part of the tensor that you are particularly interested in and want to print out. For example, for this 64x20x24x24 tensor, you can type in "0, 0" and presss enter to check the sub-tensor with shape 24x24 at coordinate (0, 0). 

### Check Tensor Value

Sometimes, developers might want to check if the tensor contains unexpected values which could be negative values, NaNs, infinities or others. To facilitate that, you can use these APIs:

```c++
template<typename ValueChecker>
std::vector<std::vector<int>> check_value(const ValueChecker& checker,
		bool interactive = false, std::string tag = "");
// OR
std::vector<std::vector<int>> check_value(CheckerType ct,
		bool interactive = false, std::string tag =  "");
```

In the first API, `ValueChecker checker` is a bool lambda function that takes in a single parameter which is of the same data type as the tensor.  For example:

```c++
// use the same DType as in the tensor object
[] (DType x) {return x == 0};
```

This checker is called on every value within the tensor. The return of the API is a `vector` of all the coordinates where the checker evaluates to `true`. The coordinates are themselves represented by `vector<int>`. If you set `interactive` to true, you will set a "break point" and enter a loop that asks for commands. This is similar to `interactive_print()`. You can type "p" to print the coordinates, "b" to break from the loop, and "s" to skip all future "break points" in `interactive_print()`. You can also specify a coordinate to print only a part of the tensor or type "e" to print out the entire tensor.  Just like `interactive_print()`, this this interactive screen is also properly locked.

![tensor_inspector_value_check](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/faq/tensor_inspector_tutorial/tensor_inspector_value_check.png)

Also, there are a bunch of built-int value checkers. Refer to the Enum below:

```c++
enum  CheckerType {
	NegativeChecker, // check if negative
	PositiveChecker, // check if positive
	ZeroChecker, // check for zero
	NaNChecker, // check if for NaN, will always return false if DType is not a float type
	InfChecker, // check for infinity, will always return false if DType is not a float type
	PositiveInfChecker, // check for positive infinity,
						// will always return false if DType is not a float type
	NegativeInfChecker, // check for nagative infinity,
						// will always return false if DType is not a float type
	FiniteChecker, // check if finite, will always return false if DType is not a float type
	NormalChecker, // check if it is neither infinity nor NaN
	AbnormalChecker, // chekck if it is infinity or nan
};
```

Remember the second API?

```c++
std::vector<std::vector<int>> check_value(CheckerType ct,
		bool interactive = false, std::string tag =  "");
```

You can simply pass in a value from `CheckerType` where you would have passed in your own lambda if you were using the first API. Note that it's the developer's responsibility to pass in a valid value checker.

### Dump Tensor Value

Sometimes, you might want to dump the tensor to a file in binary mode. Then, you might want to use a python script to further analyze the tensor value. Or, you might do that simply because a binary dump has better precision and is faster to load than the output copy-pasted from `print_string()` and loaded as a `JSON` string. Either way, you can use this API:

```c++
void dump_to_file(std::string tag);
```

This API will create a file with name  "{tag}_{visit_count}.npy", where tag is the name that we give to the call, and visit is the visit count, should the operated be called more than once.

The output format is `.npy`, version 1.0. This is the Numpy format and we can easily load it with the following code:

```
import numpy as np
a = np.load('abc_1.npy')
print(a)
```

Let's see how it runs:

![tensor_inspector_dump_to_file](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/faq/tensor_inspector_tutorial/tensor_inspector_dump_to_file.png)

Notice: in `interactive_print()`, you could also do value dumping with command "d". You will be prompted to enter the `tag` value:

![tensor_inspector_interactive_print](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/faq/tensor_inspector_tutorial/tensor_inspector_interactive_print.png)

### Test Coverage and Limitations

This utility has been tested on Mac and Ubuntu with and without CUDNN and MKLDNN. Supports for `Tensor`, `TBlob`, and `NDArray`, as well as for CPU and GPU have been manually tested. 

Currently, this utility only supports non-empty tensors and tensors with known shapes i.e. `tb_.ndim() > 0`. Also, this utility only supports dense `NDArray` objects, i.e. when the type is `kDefaultStorage`. 

