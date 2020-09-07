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

C++ Custom Operator Example and Tutorial
========================================

## Introduction

Adding new operators in MXNet requires understanding of MXNet backend operator registration and recompiling of MXNet with all its dependencies. Users can use the old Python custom operator to add new operators, but it is slow, complicated and has poor adoption rate. So our approach for adding custom operators is to enable dynamic loading of C++ custom operators compiled in external libraries at runtime.

Custom operators (CustomOp) enable users to write new operators without compiling against all of MXNet header files and dependencies. When a library containing custom operators is loaded dynamically, the operators found in the library will be registered in MXNet so that users can call those operators natively just like other built-in operators.

## Getting Started

### Have MXNet Ready

To run the following example, the build type of MXNet doesn’t matter since the custom operator doesn’t interact with the execution of other native MXNet operators.
Note that if you want to run GPU examples and write your custom operators running on GPU, you still need an MXNet CUDA build.

### Run An Example

You can start getting familiar with custom operators by running some examples provided in the `example/extensions/lib_custom_op` directory. Start with a common linear algebra operator like `gemm` (Generalized Matrix Multiplication). Go to `lib_custom_op` directory and follow these steps:

1. Run `make gemm_lib`. The Makefile will generate a dynamic library **libgemm_lib.so** compiled from `gemm_lib.cc`. This is the library you are going to load that contains everything for the custom gemm operator.
2. Run `python test_gemm.py`. It’ll first load the library compiled from step 1, find the operators, register them in the MXNet backend, then invoke the operator like a regular MXNet operator and output the result.
Below is the output when running the python `test_gemm.py` command. Notice that it loads 2 operators: `my_gemm` and `state_gemm`.

```
[19:22:02] ../src/c_api/c_api.cc:286: Found 2 operators in library
[19:22:02] ../src/c_api/c_api.cc:350: 	Op[0] my_gemm
[19:22:02] ../src/c_api/c_api.cc:350: 	Op[1] state_gemm
[19:22:02] ../src/c_api/c_api.cc:785: Found 0 partitioners in library
--------start ndarray compute---------
[[ 50.]
 [122.]]
<NDArray 2x1 @cpu(0)>
...
```

Note that you can safely ignore the `Found 0 partitioners` info as it is not related to the custom operator.

### Basic Files For A GeMM Library

* **lib_custom_op/gemm_lib.cc**: This file has a source code implementation of all required components of a custom operator, as well as the registration of the custom operator.

* **lib_custom_op/Makefile**: This file compiles `gemm_lib.cc` to a dynamic shared library named `libgemm_lib.so`. It includes the header file `include/mxnet/lib_api.h` from MXNet source code. Currently the custom operator APIs require C++11 onwards.

* **lib_custom_op/test_gemm.py**: This file calls `mx.library.load(‘libgemm_lib.so’)` to load the library containing the custom operator, invokes the operator using both NDArray and Symbol APIs, and prints outputs of the forward and backward passes. The outputs should be the same as the regular MXNet `gemm` operator.

* **include/mxnet/lib_api.h**: This file from MXNet source code is the single header file needed to include all necessary data types and function prototypes for writing a custom operator library.
You can either specify the include path in the `Makefile`, or copy the header file over to `example/extensions/lib_custom_op` folder.
Note that apart from this header, the custom operator library is independent of MXNet source.

## Writing A Custom CPU Operator Library

To build your own library containing custom CPU operator, compose a C++ source file like `myop_lib.cc`, include `lib_api.h` header file, and write your custom operator implementation with these required functions:
- `initialize` - Library Initialization Function
- `REGISTER_OP` - Operator Registration Macro
- `parseAttrs` - Attribute Parser
- `inferType` - Type Inference
- `inferShape` - Shape Inference
- `forward` - Forward Computation (can be replaced with `createOpState`, see below for details)

Then compile it to `libmyop_lib.so` dynamic library using the following command:

```bash
g++ -shared -fPIC -std=c++11 myop_lib.cc -o libmyop_lib.so -I ../../../include/mxnet
```

If you don't want to download MXNet source and choose to only use `lib_api.h` header, you can copy the header over to the same folder of `myop_lib.cc` and run:

```bash
g++ -shared -fPIC -std=c++11 myop_lib.cc -o libmyop_lib.so
```

Finally, you can write some code to load the library by specifying its absolute path and run your custom operator in any language binding.
Heres an example in Python (but C Predict API and C++ API work too):

```python
import os
import mxnet as mx
path = os.path.abspath('libmyop_lib.so')
mx.library.load(path)
mx.nd.my_op(...)
```

### Writing A Regular Custom Operator

There are several required building blocks for making a custom operator:

* [initialize](./gemm_lib.cc#L227):
    * This function is called when MXNet first loads the library. MXNet passes its version to this function when called.
    This gives the library the ability to check which version of MXNet is being used. It also provides a place where library state can be initialized.

```c++
    MXReturnValue initialize(int version)
```

* [parseAttrs](./gemm_lib.cc#L118):
    * This function specifies number of input and output tensors for the custom operator; also this is where a custom operator can validate the attributes (ie. options) specified by the user.

```c++
    MXReturnValue parseAttrs(
        const std::unordered_map<std::string, std::string>& attrs,
        int* num_in,
        int* num_out)
```


* [inferType](./gemm_lib.cc#L124):
    * This function specifies how the custom operator infers output data types using input data types.

```c++
    MXReturnValue inferType(
        const std::unordered_map<std::string, std::string>& attrs,
        std::vector<int>* intypes,
        std::vector<int>* outtypes)
```

* [inferShape](./gemm_lib.cc#L143):
    * This function specifies how the custom operator infers output tensor shape using input shape.

```c++
    MXReturnValue inferShape(
        const std::unordered_map<std::string, std::string>& attrs,
        std::vector<std::vector<unsigned int>>* inshapes,
        std::vector<std::vector<unsigned int>>* outshapes)
```

* [forward](./gemm_lib.cc#L56):
    * This function specifies the computation of the forward pass of the operator.

```c++
    MXReturnValue forward(
        const std::unordered_map<std::string, std::string>& attrs,
        std::vector<MXTensor>* inputs,
        std::vector<MXTensor>* outputs,
        const OpResource& res)
```

Also there are some optional functions you can specify:

* [backward](./gemm_lib.cc#L90) - Backward gradient function:
    * This function specifies the computation of the backward pass of the operator.

```c++
    MXReturnValue backward(
        const std::unordered_map<std::string, std::string>& attrs,
        std::vector<MXTensor>* inputs,
        std::vector<MXTensor>* outputs,
        const OpResource& res)
```

* [inferSType](./transposecsr_lib.cc#168) - Storage Type Inference:
    * This function specifies how the custom operator infers storage types for inputs and outputs.

```c++
    MXReturnValue inferSType(
        const std::unordered_map<std::string, std::string>& attrs,
        std::vector<MXTensor>* inputs,
        std::vector<MXTensor>* outputs,
        const OpResource& res)
```

* [mutateInputs](./gemm_lib.cc#L214) - Specify mutable input:
    * This function allows you to mark some inputs to be mutable inputs. It is useful when using aux parameters for BatchNorm-like operators.

```c++
    MXReturnValue mutateInputs(
        const std::unordered_map<std::string, std::string>& attrs,
        std::vector<int>* input_indices)
```

After specifying those functions, register the custom opeartor with MXNet:

* [REGISTER_OP(my_op_name)](./gemm_lib.cc#L169):
    * This macro registers the custom operator with MXNet.
    Note that you register functions for each context for `forward` and  `backward`, and here we show an example for CPU context.
    These are the minimum required functions, but you can specify additional functions as needed.

```c++
    REGISTER_OP(my_op_name)
    .setParseAttrs(parseAttrs)
    .setInferType(inferType)
    .setInferShape(inferShape)
    .setForward(forward, "cpu");
```

Let’s take a closer look at those registry functions:

* **parseAttrs**: This function takes three arguments. The 1st argument is an input, which is the attributes passed all the way from Python code. When user calls `mx.nd.my_op_name(s,t,keyword=1)`, the keyword is passed to the attributes as an entry of the map. The 2nd & 3rd arguments are outputs, and you need to set number of inputs and outputs values to those placeholders.
If the number of input and output tensors are fixed, you can use hard-coded numbers. Otherwise you can get the user-specified attributes to determine the number of inputs and outputs.

* **inferType**: This function takes three arguments. The 1st argument is the attributes (same as above). The 2nd argument is the a list of input data types corresponding to the input tensors. The 3rd argument is the placeholder for output tensor data types you need to assign.
For example, if this operator has one input and one output, and data type doesn’t change, then you can do `outtypes[0] = intypes[0]` to populate the data type.

* **inferSType**: This function takes three arguments. The 1st argument is the attributes (same as above). The 2nd argument is the a list of input storage types corresponding to the input tensors. The 3rd argument is the placeholder for output storage types you need to assign.
For example, if this operator has one input and one output, and data type doesn’t change, then you can do `outtypes[0] = intypes[0]` to populate the data type.

* **inferShape**: This function is similar to the `inferType` function, except it is used for populating the output data shapes. You need to figure out the shapes of each output tensors for this computation.
For example, if the inputs are images with shape (224,224,3) and you write a padding operator to make 10px borders for the images, then your output shape will be (234,234,3).

* **forward**: This function executes the main forward computation. It takes four arguments. The 1st argument is the attributes. The 2nd argument is the input `MXTensors` which stores all data and info of input ndarrays. The 3rd argument is the output `MXTensors`. The 4th argument is the `OpResource` object for memory allocation and other utilities. The details of `OpResource` are covered in the section below.
You can write different forward computations for each data type by doing `if(inputs[0].dtype == kFloat32)` to check the data types of tensors.
Additionally, you can use a `dltensor` tensor structure stored in the `MXTensor` as a more standardized data structure for computing.

* **backward**: This function is doing the backward gradient computation. It will be similar to the forward function. And you need to figure out the formula of the backward gradient computation.

* **mutateInputs**: This function is for marking mutable inputs. It takes two arguments. The 1st argument is the attributes. The 2nd argument is a list of input indices that are mutable among all input tensors.
For example, you can write `input_indices.push_back(1)` to mark the 2nd input tensor a mutable input.
It is useful when some inputs are auxiliary model parameters and might be altered during forward/backward computation. Remember, the index number of `input_indices` should not exceed the number of inputs.

### Writing A Stateful Custom Operator

A stateful custom operator is useful when a forward/backward call needs some data or ‘state’ from previous forward/backward calls. Normally we create a class, and make instance variables store the states used for computing or caching.

Most of the building blocks for making a stateful custom operator is the same as regular custom operator, except it’ll register `createOpState` instead of a `forward` function for the computation.

* [createOpState](./gemm_lib.cc#L204) - Create stateful operator instance:
    * This function takes two arguments. The 1st argument is attributes. The 2nd argument is a placeholder for `CustomStatefulOp` object. You must [define a class that inherits CustomStatefulOp](./gemm_lib.cc#L178) and override the forward function (optionally the backward function).
    Then you need to create an instance of your class and assign it to the placeholder. In this way, all of the forward/backward calls will use the same methods in that instance, and the instance is able to keep the state of the operator.

```c++
    MXReturnValue createOpState(
        std::map<std::string, std::string> attrs,
        CustomStatefulOp** op_inst)
```

* The operator registering function will look like this:

```c++
    REGISTER_OP(my_state_op)
    ...
    .setCreateOpState(createOpState, "cpu");
```

* Note that you will need to register each `createOpState` function specific for each context your operator supports.

## Writing A Custom GPU Operator Library

Most of the building blocks for registering GPU custom operators are the exactly same as CPU ones, except you need to specify the `"gpu"` context name when registering `forward`, `backward` or `createOpState` function.

### Run A GPU Example

For illustration purposes, we provided a `ReLU` (Rectified Linear Unit) activation operator that can run on GPU. Make sure you have installed a CUDA compatible MXNet build. Go to `lib_custom_op` directory and follow these steps: 

1. Run `make relu_lib`. The Makefile will invoke `NVCC` compiler to compile the CUDA kernel along with regular custom operator functions from `relu_lib.cu` to generate `librelu_lib.so` library.
2. Run `python test_relu.py`. It’ll register the GPU `ReLU` operator in the MXNet backend, then invoke the operator by passing an `NDArray` input with GPU context, and output the result tensor with GPU context.

### Writing A Regular GPU Custom Operator

Since most of the building blocks for registering GPU custom operators are the exactly same as CPU ones, the registering function for an operator supporting both GPU and CPU will look like this:

```c++
    REGISTER_OP(my_op)
    ...
    .setForward(forwardCPU, "cpu")
    .setForward(forwardGPU, "gpu")
    .setBackward(backwardCPU, "cpu")
    .setBackward(backwardGPU, "gpu");
```

Note that operators don’t have to support both CPU and GPU functions (can be GPU only).

After you register the forward or backward functions with `“gpu”` context, MXNet will call the appropriate forward or backward functions you just registered when the operator is invoked with GPU context.

In the `forwardGPU` function, you will specify the grid and block size and launch your CUDA kernel.
MXNet pre-allocates the memory for input and output tensors on the GPU, just like for CPU operators tensors are pre-allocated on the CPU.
As a result, you don’t need to call `cudaMemcpy` to move the tensor data to the GPU device.

```c++
    MXReturnValue forwardGPU(std::map<std::string, std::string> attrs,
                             std::vector<MXTensor> inputs,
                             std::vector<MXTensor> outputs,
                             OpResource op_res) {
        float* in_data = inputs[0].data<float>();
        float* out_data = outputs[0].data<float>();
        mx_stream_t cuda_stream = op_res.get_cuda_stream();
        ...
        my_op_forward<<<grid,block,0,cuda_stream>>>(out_data, in_data);
        ...
    }
```

Note that the `cuda_stream` object used for launching kernels is passed from MXNet backend via `OpResource` object. See below for details of `Operator Resource`. You need to compile the `lib_api.h` header file with `nvcc` if you plan to create a custom GPU operator to enable the GPU support in the APIs.  
Also, `in_data` and `out_data` are pointers to the tensor data allocated on the GPU, so you can pass them directly to your CUDA kernel.

At this point all the attribute functions for each operator (`parseAttrs`, `inferShape`, etc.) run on the CPU, including the `forwardGPU` function. The only part that will actually run on the GPU is the launched CUDA kernel function.

```c++
    __global__ void my_op_forward(float* out, float* in) {
        // code your CUDA kernel here
    }
```

### Writing A Stateful GPU Custom Operator

Recall that for stateful custom operators, you need to define a class that inherits `CustomStatefulOp` and overrides the `forward` and `backward` functions. Stateful operators are created context-aware, so you can create different classes for GPU and CPU stateful operators separately if desired. To do so, you register a createOpState function for each context separately like this

```c++
    REGISTER_OP(my_state_op_gpu)
    ...
    .setCreateOpState(createOpStateCPU, "cpu")
    .setCreateOpState(createOpStateGPU, "gpu");
```

Then you can create different classes for CPU and GPU stateful operators. MXNet will create the stateful operator instance based on the running context when the operator is invoked, and call stateful `forward` or `backward` function from the instantiated stateful operator class.

```c++
    class MyStatefulOpCPU : public CustomStatefulOp {
    public:
        explicit MyStatefulOpCPU() {}
        MXReturnValue Forward(...) {
            // code your CPU forward computational logic here
        }
        MXReturnValue Backward(...) {
            // code your CPU backward computational logic here
        }
        ~MyStatefulOpCPU() {}
    };

    class MyStatefulOpGPU : public CustomStatefulOp {
    public:
        explicit MyStatefulOpGPU() {}
        MXReturnValue Forward(...) {
            // code your GPU forward computational logic here
        }
        MXReturnValue Backward(...) {
            // code your GPU backward computational logic here
        }
        ~MyStatefulOpGPU() {}
    };

    MXReturnValue createOpStateCPU(std::map<std::string,std::string> attrs,
                                   CustomStatefulOp** op_inst) {
        *op_inst = new MyStatefulOpCPU();
        return MX_SUCCESS;
    }

    MXReturnValue createOpStateGPU(std::map<std::string,std::string> attrs,
                                   CustomStatefulOp** op_inst) {
        *op_inst = new MyStatefulOpGPU();
        return MX_SUCCESS;
    }
```

Optionally, you can use the same class for CPU and GPU, but you’ll need to check the `MXContext` type in the `MXTensors` to dispatch CPU or GPU `forward` or `backward` functions yourself to do the computation.

## Operator Resource

Most operators running in MXNet need some shared resources managed by MXNet. Custom operators also need `CPU memory allocation`, `GPU memory allocation`, and `CUDA stream` managed by MXNet backend to implement some functionalities. Those resources are provided in `OpResource` class in `forward` and `backward` functions.

1. CPU memory allocation: MXNet manages memory very carefully to reduce the memory usage and risk of memory leak. Instead of using `malloc` to obtain a temporary workspace from heap memory, it is strongly recommended to use MXNet managed memory allocation function. The `alloc_cpu(int size)` function in `OpResource` class is an API to allocate a chunk of CPU memory through MXNet, and it is safe and easy to use.

```c++
    unsigned n = inputs[1].shape[0];
    unsigned m = inputs[1].shape[1];
    void *workspace = resource.alloc_cpu(n * m * sizeof(float));
```

2. GPU memory allocation: It is almost the same as CPU memory allocation, except the API name is `alloc_gpu(int size)` and the memory chunk is located in a GPU device.

3. CUDA stream: The CUDA stream object, obtained from `get_cuda_stream()` API, helps custom operator reuse the existing MXNet CUDA stream in order to synchronize GPU running multiple kernels from multiple operators concurrently.

When you write your own custom operators, you have the option to use any of the operator resources provided above.
