# How to Create New Operations (Layers)

This note will walk you through the process of creating new MXNet operations (or layers).

We try to do our best to provide high speed operators for most common use cases. However, if you do find yourself in need of custom layers, like a novel loss for your research, you have two options:

* ~~(Deprecated) Use native language and it's matrix library (e.g. numpy in Python). This requires least effort and knowledge of MXNet. But impairs performance as it is CPU based.~~

* ~~(Deprecated) Use native language, mxnet.rtc and mxnet.ndarray. This gives you most of the performance of 3) and most of the convenience of 1), but requires more knowledge of MXNet. You can write CUDA kernels in python and compile with during runtime.~~

* 1) Use CustomOp to write new operators in frontend language (i.e. Python) that runs on cpu or gpu. Depending on your implementation, this can range from very fast to very slow.

* 2) Use C++/MShadow(CUDA). This can be difficult if you are not familiar with MXNet, mashadow or Cuda, but it will give you the best performance.

## CustomOp
Implementing an operator in Python is similar to creating one in C++ but simpler. Let's create a softmax operator for example. We start by subclassing `mxnet.operator.CustomOp` and then override a few methods:
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
Here we defined the computation for forward pass of our operator. The forward function takes a list of input and a list of output NDArrays. Here we called .asnumpy() on the input NDArray to convert it to cpu based numpy arrays for convenience.

Keep in mind that this can be very slow. If you want the best performance, keep data in NDArray format and use operations under mx.nd to do the computation.

At the end, we used CustomOp.assign to assign the resulting array y to out_data[0]. It handles assignment based on the value of req, which can be 'write', 'add' or 'null'.

Then we do the same for backward:
```python
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))
```

Softmax defines the computation of our custom operator, but we still need to define it's input/output format by subclassing mx.operator.CustomOpProp.
First we register our new operator with the name 'softmax':
```python
@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
```
Then we call our base constructor with `need_top_grad=False` be cause softmax is a loss layer and we don't need gradient input from layers above:
```python
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)
```

Then we declare our input and output
```python
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']
```
Note that list arguments declares both input and parameter and we recommend ordering them as `['input1', 'input2', ... , 'weight1', 'weight2', ...]`

Next we need to provide `infer_shape` to declare the shape of our output/weight and check the consistency of our input shapes:
```python
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []
```
The first dim of an input/output tensor is batch size. Our label is a set of integers, one for each data entry, and our output has the same shape as input. Infer_shape should always return three lists in the order inputs, outputs and auxiliary states (which we don't have here), even if one of them is empty.

Finally, we need to define a create_operator function that will be called by the backend to create an instance of Softmax:
```python
    def create_operator(self, ctx, shapes, dtypes):
        return Softmax()
```

To use your custom operator, create a mx.sym.Custom symbol with op_type being the registered name:
```python
mlp = mx.symbol.Custom(data=fc3, name='softmax', op_type='softmax')
```

The complete code for this example can be found at `examples/numpy-ops/custom_softmax.py`

## C++/MShadow(CUDA)
Please refer to [Developer Guide - SimpleOp](../system/operator_util.md) and [Developer Guide - Operators](https://mxnet.readthedocs.org/en/latest/system/operator.html) for detail.
