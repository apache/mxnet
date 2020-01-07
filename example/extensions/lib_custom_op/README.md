CustomOp Example and Tutorial
====

## Getting Started

## Have MXNet Ready:

First you should install MXNet either from compiling from source code or download from nightly build. It doesn’t matter if the build comes with CUDA or MKLDNN. The custom operator doesn’t intervene with the execution of other native MXNet operators.

## Run An Example:

You can start getting familiar with custom operator by running some examples we provide in the *example/extensions/lib_custom_op* directory. There are 2 examples: a simple 2D gemm operator, a subgraph operator, and a Makefile.

Let’s start with gemm operator. Go to that directory and follow the steps:

1. run *make gemm_lib*, the Makefile will generate a dynamic library libgemm_lib.so compiled from gemm_lib.cc. This is the library you are going to load that contains everything of the custom gemm operator.
2. run *python test_gemm.py*, and it’ll first load the above .so library, find operators,  register them in the MXNet backend, and print "Found x operators"; then invoke the operator like a regular MXNet operator and print the result.

## Basic Files For GEMM Library:

* lib_custom_op/gemm_lib.cc: This file has source code implementation of all required components of a custom operator, as well as the registration of the custom operator.

* lib_custom_op/Makefile: Compile source code to a dynamic shared library, with a header file include/mxnet/lib_api.h from MXNet source code. Currently the custom operator is compatible with C++11 onwards.

* lib_custom_op/test_gemm.py: This file calls mx.library.load(‘libgemm_lib.so’) to load custom operator, invoke the operator using both ndarray and symbol API, and print outputs of forward and backward pass. The outputs should be the same as the regular MXNet gemm operator.

## Writing Custom Operators:

## Regular Custom Operator:

There are several basic building blocks for making a (stateless) custom operator:

* parseAttrs - Attributes Parser: This function specifies number of input and output tensors for the custom operator. 

* inferType - Type Inference: This function specifies how custom operator infers output data types using input data types

* inferShape - Shape Inference: This function specifies how custom operator infers output tensor shape using input shape

* forward - Forward function: This function specifies the computation of forward pass of the operator

* REGISTER_OP(my_op_name) Macro: This macro registers custom operator to all MXNet APIs by its name, and you need to call setters to bind the above functions to the registered operator.

Also there are some operational functions you can specify:

* backward - Backward Gradient function: This function specifies the computation of backward pass of the operator

* mutateInputs - Mutate Input Mark: This function allows you to mark some inputs to be mutate inputs, useful when using aux parameters for BatchNorm-like operators

Let’s take a closer look at those registry functions:

* parseAttrs: This function takes 3 parameters. 1st parameter is an input, which is the attributes passed all the way from Python code. When user calls mx.nd.my_op_name(s,t,keyword=1), the keyword is passed to the attributes as an entry of the map. 2nd & 3rd parameters are outputs, and you need to assign num_in/num_out values to those placeholders.  If the number of input and output tensors are fixed, you can use hard-coded numbers. Otherwise you can get the keyword value to determine the num_in and num_out.

* inferType: This function takes 3 parameters. 1st parameter is the attributes. 2nd parameter is the a list of input data type enum corresponding to the data types of input tensors. 3rd parameter is the placeholder for output tensor data types you need to assign. For example, if this operator has 1 input and 1 output and data type doesn’t change, then you can do outtypes[0] = intypes[0]; to populate the data type.

* inferShape: This function is similar to inferType function, except it is used for populating the output data shapes. You need to figure out the shapes of each output tensors for this computation.

* forward: This function is doing the main forward computation. It also takes 3 parameters. 1st parameter is the attributes. 2nd parameter is the a list of input MXTensors which stores all data and info of input ndarrays. 3rd parameter is the output MXTensors. You need to do the forward computing given the input tensors and data types, and write the result back to the output tensor data pointer. Additionally you can use dltensor tensor structor stored in MXTensor as a more standardized data structure for computing.

* backward: This function is doing the backward gradient computation. It will be similar to forward function. And you need to  figure out the formula of backward.

* mutateInputs: This function is for marking mutate inputs. It takes 2 parameters. 1st parameter is the attributes. 2nd parameter is a list of  indices of mutate inputs among all input tensors. It is useful when some inputs are auxiliary model parameters and might be altered during forward/backward computation. Remember the index number of input_indices should not exceed the number of inputs.

## Stateful Custom Operator:

Stateful operator is useful when a forward/backward call needs some data or ‘state’ from the previous forward/backward call. Idiomatically we create a class and make instance variables store the state used for computing or caching.

Most of the building blocks for making stateful custom operator is the same as regular custom operator, except it’ll register *createOpState* instead of forward for the computation.

* createOpState: This function takes 2 parameters. 1st parameter is attributes. 2nd parameter is a placeholder for  CustomStatefulOp object. You must define a class that inherits CustomStatefulOp and override the forward function. Then you need to create an instance and assign it to the placeholder, in this way all the forward/backward calls will use the same methods in that instance and the instance is able to keep the state.
