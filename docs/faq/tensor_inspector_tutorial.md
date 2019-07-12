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

When developing new operators, developers need to deal with tensor objects extensively. This new utility, Tensor Inspector, mainly aims to help developers debug by providing unified interfaces to print, check, and dump the tensor value. To developers' convenience, This utility works for all the three data types: Tensors, TBlobs, and NDArrays. Also, it supports both CPU and GPU tensors.


## Usage 

This utility locates in `src/common/tensor_inspector.h`. To use it in any operator code, just include `tensor_inspector`, construct an `TensorInspector` object, and call the APIs on that object. You can run any script that uses the operator you just modified then.

The screenshot below shows a sample usage in `src/operator/nn/convolution-inl.h`.

![Screen Shot 2019-07-08 at 5 03 46 PM](https://user-images.githubusercontent.com/16669457/60850062-68690e00-a1a2-11e9-8268-033edde17aa4.png)


## Functionalities/APIs

### Create a TensorInspector Object from Tensor, TBlob, and NDArray Objects

You can create a `TensorInspector` object by passing in two things: 1) an object of type `Tensor`, `Tbob`, or `NDArray`, and 2) an `RunContext` object.

Essentially, `TensorInspector` can be understood as a wrapper class around `TBlob`. Internally, the `Tensor`, `Tbob`, or `NDArray` object that you passed in will all be converted to a `TBlob` object. The `RunContext` object is used when the the tensor is a GPU tensor; in such case, we need to use the context information to copy the data from GPU memory to CPU/main memory.

Below are the three constructors:

```c++
// Construct from Tensor object
template<typename Device, int dimension, typename DType MSHADOW_DEFAULT_DTYPE>
TensorInspector(const  mshadow::Tensor<Device, dimension, DType>& ts, const RunContext& ctx);

// Construct from TBlob object
TensorInspector(const TBlob& tb, const RunContext& ctx);

// Construct from NDArray object
TensorInspector(const NDArray& arr, const RunContext& ctx):
```

### Print Tensor Value (Static) 

To print out the tensor value in a nicely structured way,  you can use this API:

```c++
void print_string();
```

This API will print the entire tensor to `std::cout` and preserve the shape (it supports all dimensions from 1 and up). You can copy the output and interpret it with any `JSON` loader. Also, on the last line of the output you can find some useful information about the tensor. Refer to the case below, we are able to know that this is a float-typed tensor with shape 20x1x5x5.

![Screen Shot 2019-07-08 at 4 07 16 PM](https://user-images.githubusercontent.com/16669457/60848554-d8c06100-a19b-11e9-9fe0-23e79a7a371a.png)

If instead of printing the tensor to `std::cout`, you just need a `string`, you can use this API:
```c++
std::string void to_string();
```

### Interactively Print Tensor Value (Dynamic) 

When debugging, situations might occur that at compilation time, you do not know which part of a tensor to inspect. Also, sometimes, it would be nice to pause the operator control flow to “zoom into” a specific, erroneous part of a tensor multiple times until you are satisfied. In this regard, you can use this API to interactively inspect the tensor:

```c++
void  interactive_print(std::string tag =  "") {
```

This API will set a "break point" in your code, so that you will enter a loop that will keep asking you for further command. In the API call, `tag` is an optional parameter to give the call a name, so that you can identify it when you have multiple `interactive_print()` calls in different parts of your code. A visit count will tell you for how many times have you stepped into this particular "break point", should this operator be called more than once. Note that all `interactive_print()` calls are properly locked, so you can use it in many different places without issues.

![Screen Shot 2019-07-10 at 5 29 07 PM](https://user-images.githubusercontent.com/16669457/61013632-5325e800-a338-11e9-90e6-607f17d81495.png)

Refer the screenshot above, there are many useful commands available: you can type "e" to print out the entire tensor, ''d" to dump the tensor to file (see below), "b" to break from this command loop, and "s" to skip all future `interactive_print()`. Most importantly, in this screen, you can specify a part of the tensor that you are particularly interested in and want to print out. For example, for this 20x1x5x5 tensor, you can type in "0, 0" and presss enter to check the sub-tensor with shape 5x5 at coordinate (0, 0). 

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

This checker is called on every value within the tensor. The return of the API is a `vector` of all the coordinates where the checker evaluates to `true`. The coordinates are themselves represented by `vector<int>`. If you set `interactive` to true, you will set a "break point" and enter a loop that asks for commands. This is similar to `interactive_print()`. You can type "p" to print the coordinates, "b" to break from the loop, and "s" to skip all future "break points" in `interactive_print()`.  Just like `interactive_print()`, this this interactive screen is also properly locked.

![Screen Shot 2019-07-10 at 5 34 20 PM](https://user-images.githubusercontent.com/16669457/61013773-fe36a180-a338-11e9-9a2b-5f11ccc7afa7.png)

Also, there are a bunch of built-int value checkers. Refer to the Enum below:

```c++
enum  CheckerType {
	NegativeChecker, // check if is negative
	PositiveChecker, // check if is positive
	ZeroChecker, // check if is zero
	NaNChecker, // check if is NaN, will always return false if DType is not a float type
	InfChecker, // check if is infinity, will always return false if DType is not a float type
	PositiveInfChecker, // check if is positive infinity,
						// will always return false if DType is not a float type
	NegativeInfChecker, // check if is nagative infinity,
						// will always return false if DType is not a float type
	FiniteChecker, // check if is finite, will always return false if DType is not a float type
	NormalChecker, // check if is neither infinity nor NaN
	AbnormalChecker, // chekck if is infinity or nan
};
```

Remember the second API?

```c++
std::vector<std::vector<int>> check_value(CheckerType ct,
		bool interactive = false, std::string tag =  "");
```

You can simply pass in a value from `CheckerType` where you would have passed in your own lambda if you were using the first API.

### Dump Tensor Value

Sometimes, you might want to dump the tensor to a file in binary mode. Then, you might want to use a python script to further analyze the tensor value.  Or, you might do that simply because a binary dumps has better precision and is faster to load than if you copy-paste the output from `print_string()` and load it as a `JASON` string. Either way, you can use this API:

```c++
void dump_to_file(std::string tag);
```

This API will creat a file with name  "{tag}_{visit_count}.npy", where tag is the name that we give to the call, and visit is the visit count, should the operated be called more than once.

The output format is `.npy`, version 1.0. This is the Numpy format and we can easily load it with the following code:

```
import numpy as np
a = np.load('abc_1.npy')
print(a)
```

Let's see the how it runs:

![Screen Shot 2019-07-10 at 5 17 29 PM](https://user-images.githubusercontent.com/16669457/61013259-cc244000-a336-11e9-8564-a018041634f6.png)

Notice: in `interactive_print()`, you could also do value dumping with command "d". You will be prompt to enter the `tag` value:

![Screen Shot 2019-07-11 at 4 57 41 PM](https://user-images.githubusercontent.com/16669457/61092906-0f48e680-a3fd-11e9-8251-c4371cdd00ad.png)



