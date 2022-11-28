---
layout: page_category
title: A Beginner's Guide to Implementing Operators in MXNet Backend
category: faq
faq_c: Extend and Contribute to MXNet
question: How do I implement operators in MXNet backend?
permalink: /api/faq/add_op_in_backend
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

# A Beginner's Guide to Implementing Operators in MXNet Backend

## Introduction
Operators are essential elements for constructing neural networks. They define mathematical formulas
of transforming input data (tensors) to outputs. MXNet has a rich set of operators from simple ones,
such as element-wise sum, to complicated ones, such as convolution, that is
capable of constructing most of the popular neural networks. You may have noticed
that many operators implemented in MXNet have their equivalent forms in Numpy, such as
[repeat](https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html),
[tile](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html),
etc., and wonder why we could not simply use those Numpy operators in MXNet. One of the
major reasons is that we need to support both CPU and GPU computing for the operators in MXNet,
while Numpy operators do not possess GPU computing capability.
In addition, we have performed plenty of
optimizations for various components in MXNet, such as tensor data structure (`NDArray`),
execution engine, computational graph and so on, for maximizing memory and runtime efficiency.
An operator implemented under the MXNet operator framework would greatly
leverage those optimizations for exhaustive performance enhancement.

In this tutorial, we are going to practice implementing an operator using
C++ in the MXNet backend. After finishing the implementation,
we will add unit tests using Python for the operator we just implemented.

## Implementation

### An Operator Example

Let's take the [quadratic function](https://en.wikipedia.org/wiki/Quadratic_function)
as an example: `f(x) = ax^2+bx+c`. We want to implement an operator called `quadratic`
taking `x`, which is a tensor, as an input and generating an output tensor `y`
satisfying `y.shape=x.shape` and each element of `y` is calculated by feeding the
corresponding element of `x` into the quadratic function `f`.
Here variables `a`, `b`, and `c` are user input parameters.
In frontend, the operator works like this:

```python
x = [[1, 2], [3, 4]]
y = quadratic(data=x, a=1, b=2, c=3)
y = [[6, 11], [18, 27]]
```

To implement this, we first create three files: `quadratic_op-inl.h`,
`quadratic_op.cc`, and `quadratic_op.cu`. The header file's name
is prefixed by the operator name and followed by `op` and `-inl`
indicating that this is an operator implementation with inline
functions shared by CPU and GPU computing. The CPU and GPU
specific implementations reside in their own `.cc` and `.cu` files,
respectively. We normally put pure tensor related operators
(e.g. `tile`, `repeat`, etc.) under
the directory `src/operator/tensor`, and neural network operators
(e.g. `Convolution`, `Pooling`, etc.) under `src/operator/nn`.
You may have noticed that many neural network operators including
`Convolution` and `Pooling` are currently saved under `src/operator`.
We plan to move them to `src/operator/nn` for better file organization
and clearer hierarchy in the future.

Next, we are going to
1. Define the parameter struct
for registering `a`, `b`, and `c` in `quadratic_op-inl.h`.
2. Define type and shape inference functions in `quadratic_op-inl.h`.
3. Define forward and backward functions in `quadratic_op-inl.h`.
4. Register the operator using [nnvm](https://docs.tvm.ai/dev/nnvm_overview.html)
in `quadratic_op.cc` and `quadratic_op.cu` for
CPU and GPU computing, respectively.

Now let's walk through the process step by step.

### Parameter Registration
We first define `struct QuadraticParam` as a placeholder for the
parameters `a`, `b`, and `c` in `quadratic_op-inl.h`.
The struct inherits from a base template
struct named `dmlc::Parameter`, where the template argument is the derived struct
`QuadraticParam`. This technique, which is called [curiously recurring template
pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern),
achieves static polymorphism. It is similar to using a virtual function,
but without the cost associated with dynamic polymorphism.

```cpp
struct QuadraticParam : public dmlc::Parameter<QuadraticParam> {
  float a, b, c;
  DMLC_DECLARE_PARAMETER(QuadraticParam) {
    DMLC_DECLARE_FIELD(a)
      .set_default(0.0)
      .describe("Coefficient of the quadratic term in the quadratic function.");
    DMLC_DECLARE_FIELD(b)
      .set_default(0.0)
      .describe("Coefficient of the linear term in the quadratic function.");
    DMLC_DECLARE_FIELD(c)
      .set_default(0.0)
      .describe("Constant term in the quadratic function.");
  }
};
```

The function calls in the above parameter struct are self-explanatory by their names.
Note that for each parameter, we set the default value to `0.0` such that users can
skip passing 0-value parameters through the quadratic operator interface. You
can choose not to define the default value for a parameter if it is required
at runtime. Meanwhile, adding brief descriptions to the parameters enables
the documentation engine to display them on
[MXNet documentation web page]({{'/api/python/docs/api'|relative_url}}).


### Attribute Inference
Attribute inference is the process of deducing the properties of `NDArray`s
in neural networks from user provided information. Two most common attributes
of an `NDArray` are data shape and data type.
Let's take a look at the following example.
Given an input `NDArray` called `data`, you invoke the `quadratic` operator
like this: `output = mx.nd.quadratic(data, a=1, b=2, c=3)`. Before calculating
the `output` values, its shape and data type are inferred from the input
`data`'s shape and type following
the rules you defined in order to allocate memory space for the output tensor.

One important thing to note that inference functions should be capable of
performing **mutual inference**, i.e.
inferring one argument's attribute from another argument's attribute if
possible according to the definition of the operator.
This is very useful for a computational graph to deduce unknown attributes
for a neural network in symbolic programming. Users can view the computational
graph as a symbol with every element initialized for running data
throughout the neural network, including memory allocation for each tensor,
device placement for each operator, etc. Users normally just need
to provide minimum necessary information, such as input data shapes, etc.,
to the computational graph, and the graph will fill up the unknown attributes
using the attribute inference functions defined in the operators building up
the neural network.

Let's consider the following example.

```python
>>> import mxnet as mx
>>> a = mx.sym.Variable('a', shape=(2, 0))
>>> b = mx.sym.Variable('b')
>>> c = mx.sym.Variable('c', shape=(0, 3))
>>> d = a * b + b * c
>>> print d.infer_shape()
([(2L, 3L), (2L, 3L), (2L, 3L)], [(2L, 3L)], [])
```

The last line of the above code snippet is a tuple of three lists returned
by `d.infer_shape()`. The first list contains all the argument shapes
of `a`, `b`, and `c`. The second contains the output shape of `d`. The
third one represents the shapes of auxiliary states, which is not used
in this case, and thus is empty.
In this example, we only specified values for variable `a`'s first dimension
and `c`'s second dimension. The `0` in shape `(2, 0)` indicates that the size
of the second dimension is unknown, same meaning for shape `(0, 3)`.
However, the symbol `d` still successfully inferred the shapes
for all the variables and final output. This is a result of mutual
inference. In MXNet, the whole process can be interpreted as this:
1. `a` and `b` are combined via an element-wise multiplication operator,
so the shapes of `a` and `b` are same and `b`'s first dimension size is `2`.
2. `b` and `c` are combined via an element-wise multiplication operator too,
so the shapes of `b` and `c` are same and `b`'s second dimension size is `3`.
3. Now `b`'s shape is completely known, so `a` and `c` missing dimension sizes
are known as well.
4. `d` is a result from adding `a * b` and `b * c`, so d should also
have the same shape as `b`.

The above four steps illustrate how shape inference logic works in MXNet.
It is actually implemented in the shape inference functions of the operators for
element-wise multiplication and addition.

For our `quadratic` operator, shape inference possesses quite similar logic.

```cpp
inline bool QuadraticOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}
```

Here are a few things to note about the above function:

1. `attrs` contains parameters `a`, `b`, and `c` from user input.
It's not used here since we don't rely on that information for shape inference.
2. `in_attrs` is a vector containing all input shapes. Since there is
only one input argument for operator `quadratic`, we used macro `CHECK_EQ`
to assert when the vector's size is wrong.
3. `out_attrs` is a vector containing all output shapes. We also used
`CHECK_EQ` to verify the size of the vector since there is only one output.
4. We called macro `SHAPE_ASSIGN_CHECK` twice for mutual inference. One for
inferring the output shape from the input shape, the other one is for inferring
the input shape from the output shape.
If there are any unequal non-zero values in the same
dimension of two shapes, such as `(2, 3)` and `(3, 3)`, the macro would throw an
exception with an error message for shape inference.
5. At the end of the function body, we checked whether the output shape
is completely known by testing whether the shape is not empty and
the shape's size is greater than `0`. Note that in MXNet, an empty shape
means that the shape is unknown, and
a `0` in a shape means that the size of that dimension is unknown. In both
situations, the missing shape information must
be inferred from other shapes. If it cannot be inferred,
the function should return `false` to notify the caller about shape inference failure.
6. MXNet provides a convenience function implementing the logic of mutual inference
for general element-wise operators with the following interface. Users can
instantiate this function with `n_in=1` and `n_out=1` to replace the above
function `QuadraticOpShape` in operator registration (explained later).
The function `QuadraticOpShape` posted here is for the purpose of illustration only.

```cpp
template<int n_in, int n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs);
```

The same logic goes for data type inference. We will leave the analysis of
the following code sample to users. Note that `-1` means the data type
is unknown and must be inferred from other input or output data types.

```cpp
inline bool QuadraticOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}
```

Again, MXNet provides the following convenience function for mutual
type inference of element-wise operators. Users can use that
in operator registration (explained later).

```cpp
template<int n_in, int n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs);
```

### Forward Function
Forward function defines the operator's behavior in the forward pass
of neural networks. For our `quadratic` operator, it simply implements
the logic of running a tensor through the quadratic function by performing
a few element-wise operations. The forward function's signature is fixed
in MXNet as follows:

```cpp
void (const nnvm::NodeAttrs& attrs,
      const OpContext& ctx,
      const std::vector<TBlob>& inputs,
      const std::vector<OpReqType>& req,
      const std::vector<TBlob>& outputs);
```

We first paste the whole forward function code here
and then go through it line by line.


{% raw %}

```cpp
template<typename xpu>                                                        // 1
void QuadraticOpForward(const nnvm::NodeAttrs& attrs,                         // 2
                        const OpContext& ctx,                                 // 3
                        const std::vector<TBlob>& inputs,                     // 4
                        const std::vector<OpReqType>& req,                    // 5
                        const std::vector<TBlob>& outputs) {                  // 6
  CHECK_EQ(inputs.size(), 1U);                                                // 7
  CHECK_EQ(outputs.size(), 1U);                                               // 8
  CHECK_EQ(req.size(), 1U);                                                   // 9
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();                            // 10
  const TBlob& in_data = inputs[0];                                           // 11
  const TBlob& out_data = outputs[0];                                         // 12
  const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);      // 13
  using namespace mxnet_op;                                                   // 14
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {                           // 15
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {                               // 16
      Kernel<quadratic_forward<req_type>, xpu>::Launch(                       // 17
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),  // 18
          param.a, param.b, param.c);                                         // 19
    });                                                                       // 20
  });                                                                         // 21
}                                                                             // 22
```

{% endraw %}

- Line 1: `xpu` stands for a generic device type so that the function can be instantiated
for both CPU and GPU computing using concrete types `cpu` and `gpu`. The instantiation happens
at the time when the operator is registered in `.cc` and `.cu` files.
- Line 2: `attrs` is a node attribute containing the user input parameters `a`, `b`, and `c`.
Here the node represents a placeholder for the operator in the whole computational graph for
the neural network.
- Line 3: `ctx` holds something called `stream` for
serializing asynchronous executions. Let's consider
this example for understanding the functionality of `stream`.
We want to launch several GPU kernels with the same `stream` from CPU.
Even though the launching operation is non-blocking, the `stream` guarantees
that the kernels execute in the same order on GPU as they are launched from CPU.
- Line 4: `inputs` is a vector of input tensors (only one input tensor
for the `quadratic` operator).
- Line 5: `req` is a vector of `OpReqType` values. Each value defines
the way of writing calculated values to the output tensors.
Therefore, the number of `req`s must be the same as the number of output tensors.
MXNet currently supports three types of `req` in frontend: `null`, `write`, and `add`.
`null` means skipping calculating the corresponding output tensor,
`write` means overwriting the values in the output tensor with the ones
calculated by this operator, and `add` means adding the calculated values
to the existing ones in the output tensor. Note that `null` and `add` are usually
seen in backward passes. The former is for skipping calculating
the gradients of un-learnable parameters (such as index arrays),
and the latter is for accumulating gradients throughout networks.
- Line 6: `outputs` is a vector of output tensors (only one
output tensor for the `quadratic` operator).
- Lines 7-9: Verify that the size of each vector is expected.
Otherwise, stop moving forward and print error message.
- Line 10: Get the `stream` from the `ctx` for launching kernels.
- Lines 11-12: Define the references of the input and output tensors
for later coding convenience. Note that `TBlob` can be understood
as a uniform data structure for tensors of various dimensions, such
that tensors of different dimensions can be put in a homogeneous container,
such as `std::vector` and `std::list`. You can still
get tensors of desired dimensions from a `TBlob` object through
the interface `get_with_shape`.
- Line 13: Get user input parameters from the node attribute.
- Lines 15-21: This is the place where the mathematical formula of the operator
is implemented. The macros `MSHADOW_TYPE_SWITCH` and `MXNET_ASSIGN_REQ_SWITCH` enable
the code block to work for all the supported data types and `req` types in MXNet.
Inside the inner-most macro, we launch the kernel for calculating
the output tensor such that each thread takes an element from
the input tensor, feeds it into the quadratic function, and assigns
the output element to the output tensor based on `req` type. Note that
`Kernel::Launch` serves as a universal interface for launching
parallel computation on both CPU and GPU. This allows most of
the simple operators to share the same piece of code for CPU and GPU as
parallelization approaches are often identical on both types of devices.
The kernel function is defined as the following, where the function
`Map` is executed by each thread for each input element. The `out_data.Size()`,
in the `Kernel::Launch` function corresponds to the factor by which the
workload will get parallelized among the different threads, which here
corresponds to the size of the output array. To explain a little
bit more on the two macros used in the kernel struct: (1) `MSHADOW_XINLINE` is
a consolidated macro for inlining functions compiled by both CPU and GPU
compilers. It enables CPU and GPU computing to share the same piece of code.
(2) `KERNEL_ASSIGN` is a macro for unifying the statements of different `req`s
into the same line of code. It's named `KERNEL_ASSIGN` because we call
the code blocks running parallel computation kernels.
On CPUs, the kernels are normally wrapped by the OpenMP `parallel` directive;
while on GPUs, they are the kernel functions launched by CUDA library.

```cpp
template<int req>
struct quadratic_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const float a, const float b, const float c) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i] * (a * in_data[i] + b) + c);
  }
};
```

### Backward Function
Backward functions play the role of propagating derivatives of loss function
with respect to the outputs of the last layer throughout the network to the first
layer. The whole process is often known as backward propagation. We are not
going to delineate the principle of backward propagation here since users can find
great details covered in other resources, such as
[CS231n](https://cs231n.github.io/optimization-2/) and
[How the backgropagation algorithm works](https://neuralnetworksanddeeplearning.com/chap2.html).
The problem we are going to solve here for the `quadratic` operator is that
given a tensor representing the gradient of the loss function with respect
to the output of the operator, calculate the gradient with respect to
the input of the operator. There is no need to calculate the derivatives
of loss function with respect to user input parameters `a`, `b`, and `c`
since they are not learnable parameters in the network. To formulate the problem:
given `dL/dy` and `y = a*x^2 + b*x + c`, where `L` represents the loss function and
`y` stands for the output of the quadratic tensor, we need to solve for
`dL/dx`. Using the chain-rule, it is obvious to find that

```
dL/dx = dL/dy * dy/dx = dL/dy * (2*a*x + b).
```

The above equation indicates that `dL/dx` depends on the gradient
of the output tensor and value of the input tensor.
The backward function's signature is the same as the forward function's.
With the aforementioned information in mind,
let's breakdown the following backward function line by line.

{% raw %}

```cpp
template<typename xpu>                                                       // 1
void QuadraticOpBackward(const nnvm::NodeAttrs& attrs,                       // 2
                         const OpContext& ctx,                               // 3
                         const std::vector<TBlob>& inputs,                   // 4
                         const std::vector<OpReqType>& req,                  // 5
                         const std::vector<TBlob>& outputs) {                // 6
  CHECK_EQ(inputs.size(), 2U);                                               // 7
  CHECK_EQ(outputs.size(), 1U);                                              // 8
  CHECK_EQ(req.size(), 1U);                                                  // 9
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();                           // 10
  const TBlob& out_grad = inputs[0];                                         // 11
  const TBlob& in_data = inputs[1];                                          // 12
  const TBlob& in_grad = outputs[0];                                         // 13
  const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);     // 14
  using namespace mxnet_op;                                                  // 15
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {                          // 16
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {                              // 17
      Kernel<quadratic_backward<req_type>, xpu>::Launch(                     // 18
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),  // 19
          in_data.dptr<DType>(), param.a, param.b);                          // 20
    });                                                                      // 21
  });                                                                        // 22
}                                                                            // 23
```

{% endraw %}

- Lines 1-6: Backward function has the same signature as forward function.
- Lines 7-9: Check the sizes of the function arguments. One thing to note
that since the gradient of the input depends on both the gradient of the output and
the input tensor itself, `inputs` must contain two `TBlob` objects.
- Line 10: Get the `stream` of the context for serializing asynchronous executions.
- Lines 11-13: Convenience reference variables for later use. We name `out_grad`
as the gradient of the operator output, `in_data` as the input of the operator,
and `in_grad` as the gradient of the operator input.
- Line 14: Get the parameter object of `QuadraticParam`.
- Lines 16-22: Same as in the forward function, this is where parallel
computation for `in_grad` happens. The struct `quadratic_backward` implements
the formula of calculating each element of `in_grad` by one thread as the following.

```cpp
template<int req>
struct quadratic_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
                                  const DType* in_data, const float a, const float b) {
    KERNEL_ASSIGN(in_grad[i], req, out_grad[i] * (2 * a * in_data[i] + b));
  }
};
```

### Operator Registration
So far, we have implemented necessary data structure and functions for the operator `quadratic`.
Now let's register them using `nnvm` to expose the operator `quadratic`
to frontend. Users can consider the registration process as creating the operator object
instance, saving it in the operator manager (a singleton),
and setting attributes for the operator instance.

The following code is from `quadratic_op.cc`, which is responsible
for registering the operator working on CPU.

{% raw %}

```cpp
DMLC_REGISTER_PARAMETER(QuadraticParam);                                           // 1

NNVM_REGISTER_OP(quadratic)                                                        // 2
.describe(R"code(This operators implements the quadratic function:                 // 3
.. math::

    f(x) = ax^2+bx+c

where :math:`x` is an input tensor and all operations
in the function are element-wise.

Example:

  .. code-block:: python
     :emphasize-lines: 1,3
     x = [[1, 2], [3, 4]]
     y = quadratic(data=x, a=1, b=2, c=3)
     y = [[6, 11], [18, 27]]

)code" ADD_FILELINE)                                                               // 4
.set_attr_parser(ParamParser<QuadraticParam>)                                      // 5
.set_num_inputs(1)                                                                 // 6
.set_num_outputs(1)                                                                // 7
.set_attr<nnvm::FListInputNames>("FListInputNames",                                // 8
  [](const NodeAttrs& attrs) {                                                     // 9
    return std::vector<std::string>{"data"};                                       // 10
  })                                                                               // 11
.set_attr<nnvm::FInferShape>("FInferShape", QuadraticOpShape)                      // 12
.set_attr<nnvm::FInferType>("FInferType", QuadraticOpType)                         // 13
.set_attr<FCompute>("FCompute<cpu>", QuadraticOpForward<cpu>)                      // 14
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_quadratic"})  // 15
.set_attr<nnvm::FInplaceOption>("FInplaceOption",                                  // 16
  [](const NodeAttrs& attrs) {                                                     // 17
    return std::vector<std::pair<int, int> >{{0, 0}};                              // 18
  })                                                                               // 19
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")                        // 20
.add_arguments(QuadraticParam::__FIELDS__());                                      // 21

NNVM_REGISTER_OP(_backward_quadratic)                                              // 22
.set_attr_parser(ParamParser<QuadraticParam>)                                      // 23
.set_num_inputs(2)                                                                 // 24
.set_num_outputs(1)                                                                // 25
.set_attr<nnvm::TIsBackward>("TIsBackward", true)                                  // 26
.set_attr<FCompute>("FCompute<cpu>", QuadraticOpBackward<cpu>);                    // 27
```

{% endraw %}

- Line 1: Register the parameter struct.
- Line 2: Register an operator named `quadratic` by creating an instance
of `Op` type and save it in the operator manager and return a reference
of the just created operator object.
- Lines 3-4: Add description as an operator attribute
including examples of the operator. The documentation engine will extract
this description and display it on the documentation web page.
`emphasize-lines` is optional.
For more examples and troubleshooting with doc strings, refer to the [MXNet
developer wiki's Documentation Guide](https://cwiki.apache.org/confluence/display/MXNET/Documentation+Guide).
- Line 5: Set parameter struct parser for the operator. It is used for parsing
the parameters `a`, `b`, and `c` input from frontend.
- Line 6: Set the number of inputs for the operator.
- Line 7: Set the number of outputs for the operator.
- Lines 8-11: Defines a function generating a vector of names of
the operator input arguments. This function is used to add missing
arguments that users did not specify when creating a symbolic operator.
For example, `quad_func=mx.sym.quadratic()` is still a valid symbol
since we have added the attribute `FListInputNames` to the operator node
in the computational graph. MXNet would
add the missing argument with name `quadratic0_data`, where the prefix
`quadratic0` is the operator name appended with an index and the postfix
`data` comes from the return value of the user defined `FListInputName` function.
Users still can generate an executor for the `quad_func` like the following:
```python
quad_exe = quad_func.simple_bind(ctx=mx.cpu(), quadratic0_data=(1,))
```
- Line 12: Register shape inference function.
- Line 13: Register type inference function.
- Line 14: Register forward function.
- Line 15: Register the function for creating the node of the operator in
a backward pass. Note that we used a convenience functor struct `ElemwiseGradUseIn`.
As you can tell from the name, the registered functor creates the node for gradient computation
with dependencies on the output gradient node and input node. Similarly, there are
other three functors defined as `ElemwiseGradUseOut`, `ElemwiseGradUseInOut`,
and `ElemwiseGradUseNone` for developers' convenience. In order to add
this attribute, we also need to register a backward operator for `quadratic` with
several basic attributes, as it can share attribute inference
functions with the forward operator and is not exposed to frontend.
- Lines 16-19: This registered function implies that which output tensor can reuse
which input tensor's memory space instead of allocating a new memory space for the output.
In the operator `quadratic`, there is only one input and output, and the output can reuse
the input memory space, so we store a pair of zeros in the function return vector
indicating that `inputs[0]`'s memory space can be reused by `outputs[0]`.
Note that this function just provides a hint to the computational graph initializer.
If there are other nodes depending on the input tensor, the memory space
of the input tensor will not be overwritten by the output.
- Line 20: Define the input argument name as `data` for the operator.
- Line 21: Add user input parameters `a`, `b`, and `c` as the attributes of the operator.
- Line 22: Register an operator named `_backward_quadratic` for backward pass
of the operator `quadratic`. The underscore prefix in the operator name indicates
that this is an operator not exposed to users. The convention
of naming an internally used backward operator is prepending the prefix `_backward_`
to the corresponding forward operator name.
- Line 23: Set the parameter parser for the operator `_backward_quadratic`.
- Line 24: Set the number of inputs.
- Line 25: Set the number of outputs.
- Line 26: Add `TIsBackward` attribute for the operator. The shape and type
inference passes use this attribute to determine whether a node in the graph is a
forward or backward node.
- Line 27: Register backward function.

So far, we have acquired an operator working on CPU in frontend.
In order to register the operator working on GPUs, we just need to add the following
code to `quadratic_op.cu`. Note that forward and backward functions
are registered with attribute key `FCompute<gpu>`, rather than `FCompute<cpu>`.

```cpp
NNVM_REGISTER_OP(quadratic)
.set_attr<FCompute>("FCompute<gpu>", QuadraticOpForward<gpu>);

NNVM_REGISTER_OP(_backward_quadratic)
.set_attr<FCompute>("FCompute<gpu>", QuadraticOpBackward<gpu>);
```

### Unit Test
Now we have finished implementing the operator `quadratic` in MXNet backend.
If you use python, when you type `import mxnet as mx`, two python
functions for invoking your backend implementation are
generated on the fly: one is for imperative programming
registered as `mxnet.ndarray.quadratic` or `mx.nd.quadratic` for short;
the other one is for symbolic programming registered under
module `mxnet.symbol.quadratic` or `mx.sym.quadratic` for short.

In order to unit test it in frontend, we need to add the following code
to the python file `test_operator.py`. A typical operator implementation
tests for both the `symbol` API and the `ndarray` API. The following test
has both these tests. The imperative API test, tests for the `ndarray` API,
`mx.nd.contrib.quadratic`. The `symbol` API test, tests for the complete
functionality of the operator - the forward pass and the backward
pass. To facilitate the testing of these functionalities we use three
helper functions available in the `mxnet.test_utils` module:
 - `check_symbolic_forward`
 - `check_symbolic_backward`
 - `check_numeric_gradient`

```python
def test_quadratic_function():
    def f(x, a, b, c):
        return a * x**2 + b * x + c

    a = np.random.random_sample()
    b = np.random.random_sample()
    c = np.random.random_sample()
    data = mx.symbol.Variable('data')
    quad_sym = mx.sym.contrib.quadratic(data=data, a=a, b=b, c=c)
    for dtype in [np.float16, np.float32, np.float64]:
        for ndim in range(1, 6):
            shape = rand_shape_nd(ndim, 5)
            data_np = np.random.randn(*shape).astype(dtype)
            expected = f(data_np, a, b, c)
            backward_expected = 2 * a * data_np + b

            # check imperative forward
            output = mx.nd.contrib.quadratic(mx.nd.array(data_np), a=a, b=b, c=c)
            assert_almost_equal(output.asnumpy(),expected,
                                rtol=1e-2 if dtype is np.float16 else 1e-5,
                                atol=1e-2 if dtype is np.float16 else 1e-5)
            # check forward
            check_symbolic_forward(quad_sym, [data_np], [expected],
                                    rtol=1e-2 if dtype is np.float16 else 1e-5,
                                    atol=1e-2 if dtype is np.float16 else 1e-5)
            # check backward
            check_symbolic_backward(quad_sym, [data_np], [np.ones(expected.shape)],
                                        [backward_expected],
                                        rtol=1e-2 if dtype is np.float16 else 1e-5,
                                        atol=1e-2 if dtype is np.float16 else 1e-5)
            # check backward using finite difference
            check_numeric_gradient(quad_sym, [data_np], atol=0.001)
```

In the above test we create a `quadratic` symbol and feed it into the three
utility functions. The `check_symbolic_forward` and `check_symbolic_backward`
tests the computed values against the expected values that we pass
as an argument to the function. The `check_numeric_gradient` utility function
performs [gradient checking](http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/)
to verify the implementation for the backward function of the operator.
It will perform a perturbation on the input and calculate the response
rate of the output using the
[finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method).
Then it will compare the gradient from the backward pass with the values
from the finite difference method. All three of these tests will be successful
once the comparison satisfies user specified `rtol` and `atol` values. Here `rtol`
and `atol` expand to relative tolerance and absolute tolerance respectively. They
are used to specify how far the computed values can deviate from the expected values.
They are defined as follows

```
abs(Expected_Value - Computed_Value) < RTOL * abs(Expected_Value) + ATOL
```

For example, if `rtol` is `1e-5` and `atol` is `1e-5` and the expected value is
`1.5623145`, then the computed value should lie within the range of
`(1.562288876855, 1.562340123145)` else the test will fail. Make sure you
tune the `rtol` and `atol` values accordingly. Giving very low values for `rtol`
and `atol` will likely make the test very flaky. It is recommended that you
use the flakiness checker tool to check if the test you have written is flaky
or not. You can run the flakiness checker tool for the above test with the
following command -

```bash
python tools/flakiness_checker.py test_operator.test_quadratic_function
```

Please note that for `check_symbolic_forward` and `check_symbolic_backward` we pass
both the operator symbols and expected results for comparison, for
`check_numeric_gradient` we only pass the operator symbol, as the
`check_numeric_gradient` computes the expected value using finite difference
method. Which is why it is highly recommended to add `check_numeric_gradient`
test for every operator with backward function implemented as it eliminates
the possibility of passing incorrect expected results into `check_symbolic_backward`.


## Summary
In this tutorial, we practiced implementing the operator `quadratic` in MXNet backend
and unit testing the implementation in frontend. More specifically, we added parameter
struct for user-input parameters, walked through shape and type inference workflow,
implemented forward and backward functions, and registered the operator
using nnvm. Congratulations! You now know how to add operators.
We welcome your contributions to MXNet.

**Note**: Source code in the tutorial can be found in
[quadratic_op-inl.h](https://github.com/apache/mxnet/blob/master/src/operator/contrib/quadratic_op-inl.h),
[quadratic_op.cc](https://github.com/apache/mxnet/blob/master/src/operator/contrib/quadratic_op.cc),
[quadratic_op.cu](https://github.com/apache/mxnet/blob/master/src/operator/contrib/quadratic_op.cu),
and
[test_operator.py](https://github.com/apache/mxnet/blob/master/tests/python/unittest/test_operator.py#L6514).

## Additional Resources
- [Use TensorInspector to Help Debug Operators](./tensor_inspector_tutorial)
- [Use RTC to write CUDA kernels](./using_rtc)
