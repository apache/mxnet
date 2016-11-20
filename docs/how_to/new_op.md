# How to Create New Operations (Layers)

This topic walks through the process of creating new MXNet operations (or layers).

We've done our best to provide high speed operators for most common use cases. However, if you find yourself in need of custom layers, like a novel loss for your research, you have two options:

* Use CustomOp to write new operators in the front-end language (i.e., Python) that run on CPUs or GPUs. Depending on your implementation, this can range from very fast to very slow.

* Use C++/mshadow (CUDA). This can be difficult if you're not familiar with MXNet, mashadow, or Cuda, but it provides the best performance.

## CustomOp
Implementing an operator in Python is similar to creating one in C++, but simpler. As an example, let's create a softmax operator. Start by subclassing `mxnet.operator.CustomOp`, and then override a few methods:

 ```python
    import os
    # MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
    import mxnet as mx
    import numpy as np

    class Softmax(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            x = in_data[0].asnumpy()
            y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
            y /= y.sum(axis=1).reshape((x.shape[0], 1))
            self.assign(out_data[0], req[0], mx.nd.array(y))
 ```

We defined the computation for the forward pass of our operator. The forward function takes a list of input and a list of output NDArrays. For convenience, We called .asnumpy() on the input NDArray to convert it to CPU-based NumPy arrays.

This can be very slow. If you want the best performance, keep data in NDArray format and use operations under mx.nd to do the computation.

At the end, we used CustomOp.assign to assign the resulting array y to out_data[0]. It handles assignment based on the value of req, which can be 'write', 'add', or 'null'.

Then do the same for the backward pass:

```python
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))
```

Softmax defines the computation of our custom operator, but you still need to define its input/output format by subclassing mx.operator.CustomOpProp.
First, register the new operator with the name 'softmax':

```python
@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
```

Then, call the base constructor with `need_top_grad=False` because softmax is a loss layer and you don't need gradient input from preceding layers:

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

Note that list_arguments declares both input and parameter. We recommend ordering them as follows:  `['input1', 'input2', ... , 'weight1', 'weight2', ...]`

Next, provide `infer_shape` to declare the shape of the output/weight and check the consistency of the input shapes:

 ```python
        def infer_shape(self, in_shape):
            data_shape = in_shape[0]
            label_shape = (in_shape[0][0],)
            output_shape = in_shape[0]
            return [data_shape, label_shape], [output_shape], []
 ```
The first dim of an input/output tensor is batch size. The label is a set of integers, one for each data entry, and the output has the same shape as the input. Infer_shape should always return three lists in this order: inputs, outputs, and auxiliary states (which we don't have here), even if one of them is empty.

Finally, define a create_operator function that will be called by the back end to create an instance of softmax:

```python
    def create_operator(self, ctx, shapes, dtypes):
        return Softmax()
```

To use the custom operator, create an mx.sym.Custom symbol with op_type as the registered name:

```python
mlp = mx.symbol.Custom(data=fc3, name='softmax', op_type='softmax')
```

For the complete code for this example, see `examples/numpy-ops/custom_softmax.py`.

## C++/MShadow (CUDA)
For information, see [Developer Guide - SimpleOp](http://mxnet.io/architecture/overview.html#simpleop-the-unified-operator-api) and [Developer Guide - Operators](http://mxnet.io/architecture/overview.html#operators-in-mxnet).
