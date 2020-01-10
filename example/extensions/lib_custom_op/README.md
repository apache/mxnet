CustomOp Example and Tutorial
====

## Getting Started

### Have MXNet Ready:

First you should install MXNet either from compiling from source code or download from nightly build. It doesn’t matter if the build comes with CUDA or MKLDNN. The custom operator doesn’t interact with the execution of other native MXNet operators.

### Run An Example:

You can start getting familiar with custom operator by running some examples we provide in the **example/extensions/lib_custom_op** directory. Let’s start with gemm (Generalized Matrix Multiplication) operator, a common linear algebra operator. Go to that directory and follow the steps:

1. run `make gemm_lib`, the Makefile will generate a dynamic library **libgemm_lib.so** compiled from gemm_lib.cc. This is the library you are going to load that contains everything of the custom gemm operator.
2. run `python test_gemm.py`, and it’ll first load the above .so library, find operators,  register them in the MXNet backend, print "Found x operators"; then invoke the operator like a regular MXNet operator and output the result.

### Basic Files For Gemm Library:

* **lib_custom_op/gemm_lib.cc**: This file has source code implementation of all required components of a custom operator, as well as the registration of the custom operator.

* **lib_custom_op/Makefile**: Compile source code to a dynamic shared library, with a header file **include/mxnet/lib_api.h** from MXNet source code. Currently the custom operator is compatible with C++11 onwards.

* **lib_custom_op/test_gemm.py**: This file calls `mx.library.load(‘libgemm_lib.so’)` to load the library containing the custom operator, invoke the operator using both ndarray and symbol API, and print outputs of forward and backward pass. The outputs should be the same as the regular MXNet gemm operator.

## Writing Custom Operators:

### Regular Custom Operator:

There are several basic building blocks for making a (stateless) custom operator:

* [parseAttrs](./gemm_lib.cc#L118) - Attribute Parser:
    * This function specifies number of input and output tensors for the custom operator; also this is where a custom operator can validate the attributes (ie. options) specified by the user.

            MXReturnValue parseAttrs(
                std::map<std::string,
                std::string> attrs,
                int* num_in,
                int* num_out)


* [inferType](./gemm_lib.cc#L124) - Type Inference:
    * This function specifies how custom operator infers output data types using input data types.

            MXReturnValue inferType(
                std::map<std::string, std::string> attrs,
                std::vector<int> &intypes,
                std::vector<int> &outtypes)

* [inferShape](./gemm_lib.cc#L143) - Shape Inference:
    * This function specifies how custom operator infers output tensor shape using input shape.

            MXReturnValue inferShape(
                std::map<std::string, std::string> attrs,
                std::vector<std::vector<unsigned int>> &inshapes,
                std::vector<std::vector<unsigned int>> &outshapes)

* [forward](./gemm_lib.cc#L56) - Forward function:
    * This function specifies the computation of forward pass of the operator.

            MXReturnValue forward(
                std::map<std::string, std::string> attrs,
                std::vector<MXTensor> inputs,
                std::vector<MXTensor> outputs,
                OpResource res)

* [REGISTER_OP(my_op_name) Macro](./gemm_lib.cc#L169):
    * This macro registers custom operator to all MXNet APIs by its name, and you need to call setters to bind the above functions to the registered operator.

            REGISTER_OP(my_op_name)
            .setForward(forward)
            .setParseAttrs(parseAttrs)
            .setInferType(inferType)
            .setInferShape(inferShape);

Also there are some optional functions you can specify:

* [backward](./gemm_lib.cc#L90) - Backward Gradient function:
    * This function specifies the computation of backward pass of the operator.

            MXReturnValue backward(
                std::map<std::string, std::string> attrs,
                std::vector<MXTensor> inputs,
                std::vector<MXTensor> outputs,
                OpResource res)

* [mutateInputs](./gemm_lib.cc#L214) - Specify mutable input:
    * This function allows you to mark some inputs to be mutable inputs, useful when using aux parameters for BatchNorm-like operators.

            MXReturnValue mutateInputs(
                std::map<std::string, std::string> attrs,
                std::vector<int> &input_indices)

Let’s take a closer look at those registry functions:

* **parseAttrs**: This function takes 3 arguments. 1st argument is an input, which is the attributes passed all the way from Python code. When user calls `mx.nd.my_op_name(s,t,keyword=1)`, the keyword is passed to the attributes as an entry of the map. 2nd & 3rd arguments are outputs, and you need to set number of inputs and outputs values to those placeholders.  If the number of input and output tensors are fixed, you can use hard-coded numbers. Otherwise you can get the user-specified attributes to determine the number of inputs and outputs.

* **inferType**: This function takes 3 arguments. 1st argument is the attributes (same as above). 2nd argument is the a list of input data types corresponding to the input tensors. 3rd argument is the placeholder for output tensor data types you need to assign. For example, if this operator has 1 input and 1 output and data type doesn’t change, then you can do `outtypes[0] = intypes[0]` to populate the data type.

* **inferShape**: This function is similar to inferType function, except it is used for populating the output data shapes. You need to figure out the shapes of each output tensors for this computation.

* **forward**: This function executes the main forward computation. It also takes 4 arguments. 1st argument is the attributes. 2nd argument is the input MXTensors which stores all data and info of input ndarrays. 3rd argument is the output MXTensors. 4th argument is OpResource object for memory allocation and other utilities. Additionally you can use dltensor tensor structure stored in MXTensor as a more standardized data structure for computing.

* **backward**: This function is doing the backward gradient computation. It will be similar to forward function. And you need to  figure out the formula of backward.

* **mutateInputs**: This function is for marking mutable inputs. It takes 2 arguments. 1st argument is the attributes. 2nd argument is a list of input indices that are mutable among all input tensors. It is useful when some inputs are auxiliary model parameters and might be altered during forward/backward computation. Remember the index number of input_indices should not exceed the number of inputs.

### Stateful Custom Operator:

Stateful operator is useful when a forward/backward call needs some data or ‘state’ from previous forward/backward calls. Normally we create a class and make instance variables store the states used for computing or caching.

Most of the building blocks for making stateful custom operator is the same as regular custom operator, except it’ll register **createOpState** instead of forward function for the computation.

* [createOpState](./gemm_lib.cc#L204) - Create stateful operator instance:
    * This function takes 2 arguments. 1st argument is attributes. 2nd argument is a placeholder for CustomStatefulOp object. You must [define a class that inherits CustomStatefulOp](./gemm_lib.cc#L178) and override the forward function (optionally the backward function), then you need to create an instance of your class and assign it to the placeholder. In this way all the forward/backward calls will use the same methods in that instance, and the instance is able to keep the state of the operator.

            MXReturnValue createOpState(
                std::map<std::string, std::string> attrs,
                CustomStatefulOp** op_inst)
