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
The struct must inherit from a base template
struct `dmlc::Parameter`, where the template argument is the derived struct
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
