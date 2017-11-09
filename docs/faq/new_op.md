# How to Create New Operators (Layers)

This tutorials walks you through the process of creating new MXNet operators (or layers).
We've done our best to provide high-speed operators for most common use cases.
However, if you're engaged in research,
there's a good chance you'll want to define custom layers,
like a novel loss function. In these cases, you have two options:

* Use CustomOp to write new operators using a front-end language (e.g., Python) that run on CPUs or GPUs.
Depending on your implementation, this can range from very fast (if you only use operators under mx.nd) to very slow (if you copy out the data, using `.asnumpy()`).

* Use C++/mshadow (CUDA). This provides the best performance, but can be difficult
if you're not familiar with MXNet, mshadow, or Cuda.

## CustomOp
Implementing an operator in Python is simple.
As an example, let's create a softmax operator.
Start by subclassing `mxnet.operator.CustomOp`,
and then override a few methods:

```python
import os
import mxnet as mx
import numpy as np

class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))
```

We defined the computation for the forward pass of our operator.
The forward function takes a list of input and a list of output NDArrays.
For convenience, we called `.asnumpy()` on the first NDArray in input
and convert it to a CPU-based NumPy array.
This can be very slow. If you want the best performance,
keep data in the NDArray format and use operators under mx.nd to do the computation.

At the end, we used CustomOp.assign to assign the resulting array y to out_data[0]. It handles assignment based on the value of req, which can be 'write', 'add', or 'null'.

Then do the same for the backward pass:

```python
def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    l = in_data[1].asnumpy().ravel().astype(np.int)
    y = out_data[0].asnumpy()
    y[np.arange(l.shape[0]), l] -= 1.0
    self.assign(in_grad[0], req[0], mx.nd.array(y))
```

Softmax defines the computation of our custom operator,
but you still need to define its input/output format
by subclassing mx.operator.CustomOpProp.
First, register the new operator with the name 'softmax':

```python
@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
```

Then, call the base constructor with `need_top_grad=False`
because softmax is a loss layer and you don't need gradient input from preceding layers:

```python
def __init__(self):
    super(SoftmaxProp, self).__init__(need_top_grad=False)
```

Then declare the input and output:

```python
def list_arguments(self):
    return ['data', 'label']

def list_outputs(self):
    return ['output']
```

Note that list_arguments declares both input and parameter.
We recommend ordering them as follows:  `['input1', 'input2', ... , 'weight1', 'weight2', ...]`

Next, provide `infer_shape` to declare the shape of the output/weight
and check the consistency of the input shapes:

```python
def infer_shape(self, in_shape):
    data_shape = in_shape[0]
    label_shape = (in_shape[0][0],)
    output_shape = in_shape[0]
    return [data_shape, label_shape], [output_shape], []
```
The first axis of an input/output tensor corresponds to different examples within the batch.
The label is a set of integers, one for each data entry,
and the output has the same shape as the input.
The `infer_shape` function should always return three lists in this order:
inputs, outputs, and auxiliary states (which we don't have here),
even if one of them is empty.

Optionally, you can also define `infer_type` to declare the input and output data type of your operator. Supported types are `np.float32`, `np.float64`, `np.float16`, `np.uint8`, and `np.int32`.

```python
def infer_type(self, in_type):
    dtype = in_type[0]
    return [dtype, dtype], [dtype], []
```

Finally, define a create_operator function that will be called by the back end to create an instance of softmax:

```python
def create_operator(self, ctx, shapes, dtypes):
    return Softmax()
```

To use the custom operator, create a mx.sym.Custom symbol with op_type as the registered name:

```python
mlp = mx.symbol.Custom(data=fc3, name='softmax', op_type='softmax')
```

Please see the full code for this example [here](https://github.com/dmlc/mxnet/blob/master/example/numpy-ops/custom_softmax.py).

## C++
With MXNet v0.9 (the NNVM refactor) or later, creating new operators has become easier.
Operators are now registered with NNVM.
The following code is an example on how to register an operator (checkout [src/operator/tensor](https://github.com/dmlc/mxnet/tree/master/src/operator/tensor) for more examples):

```c++
NNVM_REGISTER_OP(abs)
.MXNET_DESCRIBE("Take absolute value of the src")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1,1>);
```

The syntax is quite simple, we register the operator with a name,
then set number of inputs and outputs.
You can register attributes with any key (`FInferShape` for example) to any operator,
without having to modify a central class interface definition.

### Operator Attribute System

One of the biggest improvements brought by NNVM is the operator attribute system.
This is like traits for types in common languages like C++.
We can register any attribute to any operator, with the syntax

``` c++
NNVM_REGISTER_OP(op-name)
.set_attr<AttributeType>("AttributeKey", CorrespondingAttributeObject);
```

These attributes can be retrieved later for various purposes.
For example, `FInferShape` is used for shape inference, `FCompute<cpu>` is used for carrying out actual computation on CPU.

As long as all attributes registered with the same key have the same type,
we can register any attributes to operators.
The more attribute an operator provides,
the more information the system can use for optimization.

### List of basic attributes

In this section, we will go through the basic attributes MXNet expect for all operators.
You can find the definition for them in the following two files:

- [nnvm/op_attr_types.h](https://github.com/dmlc/nnvm/blob/master/include/nnvm/op_attr_types.h)
- [mxnet/op_attr_types.h](https://github.com/dmlc/mxnet/blob/master/include/mxnet/op_attr_types.h)

#### Descriptions (Optional)

`.describe(comment)` adds a comment to the operator. Use `.MXNET_DESCRIBE(comment)` to add the current file name and line number to comment.

#### Attribute Parser (Optional)

Set attribute parser with `.set_attr_parser(PARSER)` where PARSER is a function with prototype `void(nnvm::NodeAttr* attrs)`. This function should parse the key-word arguments in `attrs->dict` and store the result in `attrs->parsed`.

Simple arguments can be parsed like
```c++
NNVM_REGISTER_OP(scalar_op)
.set_attr_parser(
  [](NodeAttrs* attrs) {
    attrs->parsed = std::stod(attrs->dict["scalar"]);
  })
```

The parsed arguments can then be accessed in other attribute functions with
```
double alpha = nnvm::get<double>(attrs.parsed);
```

More complex ops can use `dmlc::Parameters` and `ParamParser` (defined in operator_common.h) for parsing:

``` c++
#include <dmlc/parameter.h>
#include <operator_common.h>
struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .describe("Activation function to be applied.");
  }
};
NNVM_REGISTER_OP(Activation)
.set_attr_parser(ParamParser<ActivationParam>);
// access with:
// const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
```

#### Inputs & Outputs

Number of inputs/outputs can be set with `.set_num_inputs(n_in)` and `.set_num_outputs(n_out)`
where n_in and n_out are integers.

Alternatively, if the number of inputs/outputs is variable and depends on arguments,
you can set `n_in`/`n_out` to functions with prototype `uint32_t(const nnvm::NodeAttrs& attrs)`
that return the number of inputs/outputs based on parsed arguments.

Outputs can be made invisible to other operators by registering `FNumVisibleOutputs`
and returning an integer smaller than `n_out`.

Inputs/outputs can be named by registering `FListInputNames` and `FListOutputNames` with prototype `std::vector<std::string>(const NodeAttrs& attrs)`.


#### Argument Descriptions

Set argument descriptions with `.add_argument(name, type, comment)`.
This is necessary for operators to be properly called imperatively.

First, add NDArray arguments `num_inputs` times with type "NDArray"
or one time with type "NDArray[]" for ops with variable length inputs.

Then add key-word arguments with proper type (float, string, etc).
Operators that parse key-word arguments with `dmlc::Parameter`
can add argument descriptions in bulk with `.add_arguments(ActivationParam::__FIELDS__())`
(NDArray arguments still need to be manually added with type "NDArray").

#### FInferShape or TIsBackward (for Backward Only Ops)

Normally operators need to have `FInferShape` with prototype `bool(const nnvm::NodeAttrs& attrs, std::vector<TShape> *in_attrs, std::vector<TShape> *out_attrs)`. `FInferShape` fills unknown shapes (`shape.ndim() == 0`) in in_attrs/out_attrs based on known shapes in in_attrs/out_attrs. Use `ElemwiseShape<n_in, n_out>` for simple operators with uniform shapes.

Operators that are only used for a backward pass can instead register `.set_attr<nnvm::TIsBackward>("TIsBackward", true)`
and their shapes with be copied from the corresponding forward operators.

#### FInferType

Similar to `FInferShape`, `FInferType` fills unknown types (-1) based on known types. Use `ElemwiseType<n_in, n_out>` for simple operators with uniform types. Operators that registered `TIsBackward` don't need to register this.


#### FInplaceOption (Optional)

`FInplaceOption` with prototype `std::vector<std::pair<int, int> >(const NodeAttrs& attrs)`
specifies which input/output pairs can be computed in-place
and share memory with each other.
Each pair (i, j) in the returned list means
that the i-th input can share memory with the j-th output.


#### FGradient (Optional for imperative use, required for symbolic use)

If an operator has gradient, it can be described with `FGradient` with prototype

``` c++
std::vector<nnvm::NodeEntry>(const nnvm::NodePtr& n,
                             const std::vector<nnvm::NodeEntry>& ograds)
```

Use utility functions `ElemwiseGradUseIn{op_name}`, `ElemwiseGradUseOut{op_name}`, `ElemwiseGradUseNone{op_name}`  for ops that need corresponding forward op's input,
output or nothing to calculating gradient.

For more complicated patterns, use `MakeGradNode(op_name, n, heads, dict)` to create gradient entries,
where heads are input entries to the backward op, composed from ograds and n->inputs.

#### FCompute\<xpu\>

Simple operators can register FCompute<xpu> with `.set_attr<FCompute>("FCompute<cpu>", ...)` and `.set_attr<FCompute>("FCompute<gpu>", ...)` for both CPU and (optionally) GPU computation.

FCompute has prototype

```c++
void(const nnvm::NodeAttrs& attrs,
     const OpContext& ctx,
     const std::vector<TBlob>& inputs,
     const std::vector<OpReqType>& req,
     const std::vector<TBlob>& outputs)
```

`req` has the same length as `outputs`.
Each entry of `req` specifies
how the corresponding `output` should be written to.
`OpReqType` is defined as:

```c++
enum OpReqType {
  kNullOp,
  kWriteTo,
  kWriteInplace,
  kAddTo
};
```

Normally, the `req` of all `outputs` should be `kWriteTo`,
meaning that the provided `outputs` tensor is a *raw* memory block,
so the operator should write results directly into it.
In some cases, for example, when calculating the gradient tensor,
it would be great if we could accumulate the result,
rather than directly overwrite the tensor contents
so that no extra space needs to be created each time.
In such cases, the corresponding `req` is set to `kAddTo`,
indicating that a `+=` should be used.

### Example: abs operator

```c++
NNVM_REGISTER_OP(abs)
.MXNET_DESCRIBE("Take absolute value of the src")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", UnaryCompute<cpu, mshadow_op::abs>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});              
.add_argument("data", "NDArray", "Source input")

NNVM_REGISTER_OP(_backward_abs)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, backward_grad<mshadow_op::sign> >);
```

### Legacy Operators

For the legacy (pre 0.9) way of defining operators with C++, please see:
- [Developer Guide - Operators](http://mxnet.io/architecture/overview.html#operators-in-mxnet)
- [Developer Guide - SimpleOp](http://mxnet.io/architecture/overview.html#simpleop-the-unified-operator-api)
