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
Attribute inference means the process of deducing the properties of `NDArray`s
in neural networks from user input information. Two most common attributes
of `NDArray` are shape and dtype. Let's look at the following example.
Given an input `NDArray` called `data`, you invoke the quadratic operator
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
2. `b' and `c` are combined using an element-wise multiplication operator too,
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
the function should return 'false' to notify the caller about shape inference failure.

The same logic goes for data type inference. We will leave the analysis of
the following code sample to users.
```
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
