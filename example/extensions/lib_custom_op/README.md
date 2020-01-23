CustomOp Example and Tutorial
=============================

## Introduction

Adding new operators in MXNet requires understanding of MXNet backend operator registration and recompiling of MXNet with all its dependencies. Users can use the old Python custom operator to add new operators, but it is slow, complicated and has poor adoption rate. So our approach for adding custom operators is to enable dynamic loading of C++ custom operators compiled in external libraries at runtime.

Custom operators (CustomOp) enable users to write new operators without compiling against all of MXNet header files and dependencies. When a library containing custom operators is loaded dynamically, the operators found in the library will be re-registered in MXNet so that users can call those operators natively just like other built-in operators.

## Getting Started

### Have MXNet Ready

First you should install MXNet either from compiling from source code or download from nightly build. It doesn’t matter if the build comes with CUDA or MKLDNN. The custom operator doesn’t interact with the execution of other native MXNet operators.

### Run An Example:

You can start getting familiar with custom operators by running some examples provided in the **example/extensions/lib_custom_op** directory. Start with a common linear algebra operator like `gemm` (Generalized Matrix Multiplication). Go to `lib_custom_op` directory and follow these steps:

1. Run `make gemm_lib`. The Makefile will generate a dynamic library **libgemm_lib.so** compiled from `gemm_lib.cc`. This is the library you are going to load that contains everything for the custom gemm operator.
2. Run `python test_gemm.py`. It’ll first load the above .so library, find the operators, register them in the MXNet backend, print "Found x operators", then invoke the operator like a regular MXNet operator and output the result.

### Basic Files For Gemm Library:

* **lib_custom_op/gemm_lib.cc**: This file has a source code implementation of all required components of a custom operator, as well as the registration of the custom operator.

* **lib_custom_op/Makefile**: Compile source code to a dynamic shared library, with a header file `include/mxnet/lib_api.h` from MXNet source code. Currently the custom operator is compatible with C++11 onwards.

* **lib_custom_op/test_gemm.py**: This file calls `mx.library.load(‘libgemm_lib.so’)` to load the library containing the custom operator, invokes the operator using both NDArray and Symbol APIs, and prints outputs of the forward and backward passes. The outputs should be the same as the regular MXNet `gemm` operator.

## Writing Custom Operator Library:

For building a library containing your own custom operator, compose a C++ source file like `myop_lib.cc`, include `lib_api.h` header file, and write your custom operator implementation with those essential functions:
- `initialize` - Library Initialization Function
- `REGISTER_OP` - Operator Registration Marco
- `parseAttrs` - Attribute Parser
- `inferType` - Type Inference
- `inferShape` - Shape Inference
- `forward` - Forward Computation (can be replace with `createOpState`, see below for details)

Then compile it to `libmyop_lib.so` dynamic library using the following command

    g++ -shared -fPIC -std=c++11 myop_lib.cc -o libmyop_lib.so -I ../../../include/mxnet

Finally you can write a python script to load the library and run your custom operator

    import mxnet as mx
    mx.library.load(‘libmyop_lib.so’)
    mx.nd.my_op(...)

### Writing Regular Custom Operator:

There are several essential building blocks for making a (stateless) custom operator:

* [initialize](./gemm_lib.cc#L227):
    * This function is the library initialization function necessary for any dynamic libraries. It checks if you are using a compatible version of MXNet. Note that this `version` parameter is passed from MXNet when library is loaded.

            MXReturnValue initialize(int version)

* [parseAttrs](./gemm_lib.cc#L118):
    * This function specifies number of input and output tensors for the custom operator; also this is where a custom operator can validate the attributes (ie. options) specified by the user.

            MXReturnValue parseAttrs(
                std::map<std::string,
                std::string> attrs,
                int* num_in,
                int* num_out)


* [inferType](./gemm_lib.cc#L124):
    * This function specifies how the custom operator infers output data types using input data types.

            MXReturnValue inferType(
                std::map<std::string, std::string> attrs,
                std::vector<int> &intypes,
                std::vector<int> &outtypes)

* [inferShape](./gemm_lib.cc#L143):
    * This function specifies how the custom operator infers output tensor shape using input shape.

            MXReturnValue inferShape(
                std::map<std::string, std::string> attrs,
                std::vector<std::vector<unsigned int>> &inshapes,
                std::vector<std::vector<unsigned int>> &outshapes)

* [forward](./gemm_lib.cc#L56):
    * This function specifies the computation of the forward pass of the operator.

            MXReturnValue forward(
                std::map<std::string, std::string> attrs,
                std::vector<MXTensor> inputs,
                std::vector<MXTensor> outputs,
                OpResource res)

* [REGISTER_OP(my_op_name)](./gemm_lib.cc#L169):
    * This macro registers the custom operator and its properties to MXNet NDArray and Symbol APIs by its name.

            REGISTER_OP(my_op_name)
            .setForward(forward)
            .setParseAttrs(parseAttrs)
            .setInferType(inferType)
            .setInferShape(inferShape);

Also there are some optional functions you can specify:

* [backward](./gemm_lib.cc#L90) - Backward gradient function:
    * This function specifies the computation of the backward pass of the operator.

            MXReturnValue backward(
                std::map<std::string, std::string> attrs,
                std::vector<MXTensor> inputs,
                std::vector<MXTensor> outputs,
                OpResource res)

* [mutateInputs](./gemm_lib.cc#L214) - Specify mutable input:
    * This function allows you to mark some inputs to be mutable inputs. It is useful when using aux parameters for BatchNorm-like operators.

            MXReturnValue mutateInputs(
                std::map<std::string, std::string> attrs,
                std::vector<int> &input_indices)

Let’s take a closer look at those registry functions:

* **parseAttrs**: This function takes three arguments. The 1st argument is an input, which is the attributes passed all the way from Python code. When user calls `mx.nd.my_op_name(s,t,keyword=1)`, the keyword is passed to the attributes as an entry of the map. The 2nd & 3rd arguments are outputs, and you need to set number of inputs and outputs values to those placeholders.  If the number of input and output tensors are fixed, you can use hard-coded numbers. Otherwise you can get the user-specified attributes to determine the number of inputs and outputs.

* **inferType**: This function takes three arguments. The 1st argument is the attributes (same as above). The 2nd argument is the a list of input data types corresponding to the input tensors. The 3rd argument is the placeholder for output tensor data types you need to assign. For example, if this operator has one input and one output, and data type doesn’t change, then you can do `outtypes[0] = intypes[0]` to populate the data type.

* **inferShape**: This function is similar to the `inferType` function, except it is used for populating the output data shapes. You need to figure out the shapes of each output tensors for this computation. For example, if the inputs are images with shape (224,224,3) and you write a padding operator to make 10px borders for the images, then your output shape will be (234,234,3).

* **forward**: This function executes the main forward computation. It takes four arguments. The 1st argument is the attributes. The 2nd argument is the input `MXTensors` which stores all data and info of input ndarrays. The 3rd argument is the output `MXTensors`. The 4th argument is the `OpResource` object for memory allocation and other utilities. Additionally, you can use a `dltensor` tensor structure stored in the `MXTensor` as a more standardized data structure for computing.

* **backward**: This function is doing the backward gradient computation. It will be similar to the forward function. And you need to figure out the formula of the backward gradient computation.

* **mutateInputs**: This function is for marking mutable inputs. It takes two arguments. The 1st argument is the attributes. The 2nd argument is a list of input indices that are mutable among all input tensors. It is useful when some inputs are auxiliary model parameters and might be altered during forward/backward computation. Remember, the index number of `input_indices` should not exceed the number of inputs.

### Writing Stateful Custom Operator:

A stateful custom operator is useful when a forward/backward call needs some data or ‘state’ from previous forward/backward calls. Normally we create a class, and make instance variables store the states used for computing or caching.

Most of the building blocks for making a stateful custom operator is the same as regular custom operator, except it’ll register `createOpState` instead of a `forward` function for the computation.

* [createOpState](./gemm_lib.cc#L204) - Create stateful operator instance:
    * This function takes two arguments. The 1st argument is attributes. The 2nd argument is a placeholder for `CustomStatefulOp` object. You must [define a class that inherits CustomStatefulOp](./gemm_lib.cc#L178) and override the forward function (optionally the backward function). Then you need to create an instance of your class and assign it to the placeholder. In this way, all of the forward/backward calls will use the same methods in that instance, and the instance is able to keep the state of the operator.

            MXReturnValue createOpState(
                std::map<std::string, std::string> attrs,
                CustomStatefulOp** op_inst)
