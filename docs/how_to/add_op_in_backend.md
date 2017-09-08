# A Beginner's Guide to Implementing Operators in MXNet Backend

## Introduction
Operators are essential elements for constructing neural networks. They define mathematical formulas
of transforming input data (tensors) to outputs. MXNet has a rich set of operators from simple ones,
such as element-wise sum, to complicated ones, such as convolution. You may have noticed
that many operators implemented in MXNet have their equivalent forms in Numpy, such as repeat,
tile, etc., and wonder why we could not simply use those Numpy operators in MXNet. One of the
major reasons is that we need to support both CPU and GPU computing for the operators in MXNet,
while Numpy operators do not have GPU computing capability. In addition, we have performed plenty of
optimizations on various components in MXNet, such as tensor data structure (`NDArray`), execution engine,
computational graph and so on, for maximizing memory and runtime efficiency. An operator implemented
under the MXNet operator framework is able to greatly leverage those optimizations for exhaustive
performance enhancement.

In this tutorial, we are going to practice writing an operator using C++ in the MXNet backend. After
finishing the implementation, we will add unit tests using Python testing the operator
we just implemented.

## Implementation
### An Operator Example
Let's take the [quadratic function](https://en.wikipedia.org/wiki/Quadratic_function)
as an example: `f(x) = ax^2+bx+c`. We want to implement an operator called `quadratic`
taking `x`, which is a tensor, as an input and generating an output tensor `y`
satisfying `y.shape=x.shape` and each element of `y` is calculated by feeding the
corresponding element of `x` into the quadratic function `f`.
Here variables `a`, `b`, and `c` are user-input parameters.
In frontend, the operator works like this:
```python
x = [[1, 2], [3, 4]]
y = quadratic(data=x, a=1, b=2, c=3)
y = [[6, 11], [18, 27]]
```
To implement this, we first create three files: `quadratic_op-inl.h`,
`quadratic_op.cc`, and `quadratic_op.cu`. Then we are going to
1. Define the paramter struct
for registering `a`, `b`, and `c` in `quadratic_op-inl.h`.
2. Define type and shape inference functions in `quadratic_op-inl.h`.
3. Define forward and backward functions in `quadratic_op-inl.h`.
4. Register the operator through the [nnvm](https://github.com/dmlc/nnvm)
interface `quadratic_op.cc` for CPU computing and
`quadratic_op.cu` for GPU computing.

Now let's walk through the process step by step.

### Parameter Registration
We first define `struct QuadraticParam` as a placeholder for user-input
parameters `a`, `b`, and `c` in `quadratic_op-inl.h`.
The struct inherits from a base template
struct called `dmlc::Parameter`, where the template argument is the derived struct
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

The function calls in the above parameter struct are self-explanatory. Note that
for each parameter, we set the default value to 0.0 such that users can
skip passing 0-value parameters to the quadratic operator interface. You
can choose not to define the default value for a parameter if it is required
at runtime. Meanwhile, adding brief descriptions to the parameters enables
the documentation build to display them on the
[MXNet documentation web page](https://mxnet.incubator.apache.org/api/python/index.html).

### Attribute Inference
Attribute inference is the process of deducing the properties of `NDArray`s
in neural networks from user provided information. Two most common attributes
of an `NDArray` are shape and dtype. Let's take a look at the following example.
Given an input `NDArray` called `data`, you invoke the `quadratic` operator
like this `output = mx.nd.quadratic(data, a=1, b=2, c=3)`. Before calculating
the `output` values, its shape and dtype are inferred from the input
`data`'s shape and dtype following
the rules you defined in order to allocate memory space for the output tensor.

One important thing to note that inference functions should be capable of
performing mutual inference, for example,
inferring input shape from output shape, inferring one argument's shape
from another argument, etc. This is useful in building computational graphs
for neural networks. Let's consider the following example.
```python
>>> import mxnet as mx
>>> a = mx.sym.Variable('a', shape=(2, 0))
>>> b = mx.sym.Variable('b')
>>> c = mx.sym.Variable('c', shape=(0, 3))
>>> d = a * b + b * c
>>> print d.infer_shape()
([(2L, 3L), (2L, 3L), (2L, 3L)], [(2L, 3L)], [])
```
In this example, we only specified values for `a`'s first dimension
and `c`'s second dimension. The `0` in shape `(2, 0)` means the size
of the second dimension is unknown, same for shape `(0, 3)`.
However, the symbol `d` still successfully inferred the shapes
for all the variables and final output. This is a result of mutual
inference. In MXNet, the whole process can be interpreted as this:
1. `a` and `b` are combined using an element-wise multiplication operator,
so the shapes of `a` and `b` are same and `b`'s first dimension size is `2`.
2. `b` and `c` are combined using an element-wise multiplication operator too,
so the shapes of `b` and `c` are same and `b`'s second dimension size is `3`.
3. Now `b`'s shape is completely known, so `a` and `c` missing dimension sizes
are known as well.
4. `d` is a result from adding `a * b` and `b * c`, so d should also
have the same shape as `b`.

The above four steps illustrate how shape inference works in MXNet. It is the
logic implemented in the shape inference functions of operators
element-wise multiplication and addition.

For our quadratic operator, shape inference has the similar logic.
```cpp
inline bool QuadraticOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).Size() != 0U;
}
```
Here are a few things to note about the above function:

1. `attrs` contains parameters `a`, `b`, and `c` from user input.
It's not used here since we don't need that information for shape inference.
2. `in_attrs` is a vector containing all input shapes. Since there is
only one input argument for operator `quadratic`, we use macro `CHECK_EQ`
to assert when the vector's size is wrong.
3. `out_attrs` is a vector containing all output shapes. We also use
`CHECK_EQ` to verify the size of the vector since there is only one output.
4. We called macro `SHAPE_ASSIGN_CHECK` twice for mutual inference. One for
inferring output shape from input shape, the other one is for inferring
input shape from output shape. If there are any unequal non-zero values in the same
dimension of two shapes, such as (2, 3) and (3, 3), the macro would throw
exception to generate error message for shape inference.
5. At the end of the function body, we check whether the output shape
is completely known by comparing whether its size is greater than 0. If not,
the function should return `false` to notify the caller about shape inference failure.
6. MXNet provides a convenience function implementing the logic of mutual inference
for general element-wise operators with the following interface. Users can
instantiate this function with `n_in=1` and `n_out=1` to replace the above
function `QuadraticOpShape` in operator registration (explained later).
The function `QuadraticOpShape` is implemented here for illustration purpose only.
```cpp
template<int n_in, int n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs);
```

The same logic goes for data type inference. We will leave the analysis of
the following code sample to users. Note that `-1` means the data type
is unknown and must be inferred from other arguments or outputs.
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
in operator registration (discussed later).
```cpp
template<int n_in, int n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs);
```

### Forward Function
Forward function defines the operator behavior in the forward pass
of neural networks. For our `quadratic` operator, it simply implements
the logic of running a tensor through the quadratic function performing
a few element-wise operations. We paste the whole forward function code here
and let's go through it line by line.
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
- Line 1: `attrs` contains the user input parameters `a`, `b`, and `c`
- Line 2: `ctx` holds the `stream` for serializing asynchronous executions.
For example, launching GPU kernels from CPU is an asynchronous operation.
The `stream` guarantees that the kernels of the same `stream` execute
in the same order on GPU as they are launched.
- Line 3: `inputs` is a vector of input tensors (only one input tensor
for the `quadratic` operator).
- Line 4: `req` is a vector of `OpReqType` values. Each value defines
the way of writing calculated values to the output tensors.
Therefore, the number of `req`s must be same as the number of output tensors.
MXNet currently supports three types of `req`: 'null', 'write', and 'add'.
`null` means skipping calculating the corresponding output tensor,
`write` means overwriting the values in the output tensor with the ones
calculated from this operator, and `add` means adding the calculated values
to the existing ones in the output tensor.
- Line 5: `outputs` is a vector of output tensors (only one
output tensor for the `quadratic` operator).
- Lines 7-9: Verify that the size of each vector is expected.
Otherwise, stop moving forward and output error message.
- Line 10: Get the `stream` from the `ctx` for launching kernels.
- Lines 11-12: Define the references of input tensor and output tensor
for later coding convenience. Note that `TBlob` can be understood
as a uniform data structure for tensors of various dimensions, so
that tensors of different dimensions can be put in a homogeneous container,
such as `std::vector`, `std::list`, and so on. You can still
get tensors of desired dimensions from a `TBlob` object through
the interface `get_with_shape`.
- Line 13: Get user-input parameters from the node attribute.
- Lines 15-21: This is the place where the formula of the operator is implemented.
The two macros `MSHADOW_TYPE_SWITCH` and `MXNET_ASSIGN_REQ_SWITCH` enable
the code block to work for all the supported data types and `req` types in MXNet.
Insider the inner-most macro, we launch the kernel for calculating
the output tensor such that each thread takes an element from
the input tensor, feed it into the quadratic function, and assign
the output element to the output tensor based on `req`. Note that
`Kernel::Launch` serves as a universal interface for launching
parallel computation on both CPU and GPU. This allows most of
the simple operators to share the same piece of code for CPU and GPU as
parallelization approaches are identical on both types of devices.
The kernel function is defined as the following, where the function
`Map` is executed by each thread for each input element.
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
with respect to parameters in the last layer throughout the network to the first
layer. The whole process is the so-called backward propagation. We are not
going to delineate the principle of backward propagation here since users can find
great details covered in other resources, such as
[CS231n](http://cs231n.github.io/optimization-2/) and
[How the backgropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html).
The problem we are going to solve here for the `quadratic` operator is that
given a tensor representing the gradient of the loss function with respect
to the output of the operator, calculate the gradient with respect to
the input of the operator. There is no need to calculate the derivatives
of loss function with respect to user-input parameters `a`, `b`, and `c`
since they are not learnable parameters in the network. To formulate the problem:
given `dL/dy` and `y = a*x^2 + b*x + c`, where `L` represents the loss function and
`y` stands for the output of the quadratic tensor, we need to solve for
`dL/dx`. Using chain-rule, it is obvious to find that
```
dL/dx = dL/dy * dy/dx = dL/dy * (2*a*x + b).
```
The above equation indicates that `dL/dx` depends on the gradient
of the output tensor and the input tensor. With this information in mind,
let's breakdown the following backward function line by line.
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
- Lines 1-6: Backward function has the same signature as forward function.
- Lines 7-9: Verifying the sizes of the function arguments. One thing to note
that since the gradient of the input depends on both the gradient of the output and
the input tensor itself, `inputs` must contain two `TBlob`s.
- Line 10: Get the stream of the context for serializing asynchronous executions.
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
Now let's register them through `nnvm` interfaces to expose the operator `quadratic`
to frontend. The following code is from `quadratic_op.cc`. The `.cc` file is responsible
for registering the operator working on CPU.
```cpp
DMLC_REGISTER_PARAMETER(QuadraticParam);                                           // 1

NNVM_REGISTER_OP(quadratic)                                                        // 2
.describe(R"code(This operators implements the quadratic function:                 // 3
.. math::

    f(x) = ax^2+bx+c

where :math:`x` is an input tensor and all operations
in the function are element-wise.

Example::
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
