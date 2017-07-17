
# Creating custom operators with numpy

In this tutorial, we will learn how to build custom operators with numpy in python. We will go through two examples:
- Custom operator without any `Parameter`s
- Custom operator with `Parameter`s

Custom operator in python is easy to develop and good for prototyping, but may hurt performance. If you find it to be a bottleneck, please consider moving to a C++ based implementation in the backend.



```python
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
```

## Parameter-less operators

This operator implements the standard sigmoid activation function. This is only for illustration purposes, in real life you would use the built-in operator `mx.nd.relu`.

### Forward & backward implementation

First we implement the forward and backward computation by sub-classing `mx.operator.CustomOp`:


```python
class Sigmoid(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        x = in_data[0].asnumpy()
        y = 1.0 / (1.0 + np.exp(-x))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        y = out_data[0].asnumpy()
        dy = out_grad[0].asnumpy()
        dx = dy*(1.0 - y)*y
        self.assign(in_grad[0], req[0], mx.nd.array(dx))
```

### Register custom operator

Then we need to register the custom op and describe it's properties like input and output shapes so that mxnet can recognize it. This is done by sub-classing `mx.operator.CustomOpProp`:


```python
@mx.operator.register("sigmoid")  # register with name "sigmoid"
class SigmoidProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SigmoidProp, self).__init__(True)

    def list_arguments(self):
        #  this can be omitted if you only have 1 input.
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        """Calculate output shapes from input shapes. This can be
        omited if all your inputs and outputs have the same shape.

        in_shapes : list of shape. Shape is described by a tuple of int.
        """
        data_shape = in_shapes[0]
        output_shape = data_shape
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return Sigmoid()
```

### Example Usage

We can now use this operator by calling `mx.nd.Custom`:


```python
x = mx.nd.array([0, 1, 2, 3])
# attach gradient buffer to x for autograd
x.attach_grad()
# forward in a record() section to save computation graph for backward
# see autograd tutorial to learn more.
with autograd.record():
    y = mx.nd.Custom(x, op_type='sigmoid')
print(y)
```

```python
# call backward computation
y.backward()
# gradient is now saved to the grad buffer we attached previously
print(x.grad)
```

## Parametrized Operator

In the second use case we implement an operator with learnable weights. We implement the dense (or fully connected) layer that has one input, one output, and two learnable parameters: weight and bias.

The dense operator performs a dot product between data and weight, then add bias to it.

### Forward & backward implementation


```python
class Dense(mx.operator.CustomOp):
    def __init__(self, bias):
        self._bias = bias

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        weight = in_data[1].asnumpy()
        y = x.dot(weight.T) + self._bias
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0].asnumpy()
        dy = out_grad[0].asnumpy()
        dx = dy.T.dot(x)
        self.assign(in_grad[0], req[0], mx.nd.array(dx))
```

### Registration


```python
@mx.operator.register("dense")  # register with name "sigmoid"
class DenseProp(mx.operator.CustomOpProp):
    def __init__(self, bias):
        super(DenseProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self._bias = float(bias)

    def list_arguments(self):
        return ['data', 'weight']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = in_shapes[1]
        output_shape = (data_shape[0], weight_shape[0])
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape, weight_shape), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return Dense(self._bias)
```

### Use CustomOp together with Block

Parameterized CustomOp are ususally used together with Blocks, which holds the parameter.


```python
class DenseBlock(mx.gluon.Block):
    def __init__(self, in_channels, channels, bias, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self._bias = bias
        self.weight = self.params.get('weight', shape=(channels, in_channels))

    def forward(self, x):
        ctx = x.context
        return mx.nd.Custom(x, self.weight.data(ctx), bias=self._bias, op_type='dense')
```

### Example usage


```python
dense = DenseBlock(3, 5, 0.1)
dense.initialize()
x = mx.nd.uniform(shape=(4, 3))
y = dense(x)
print(y)
```
