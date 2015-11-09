# How to Create New Operations (Layers)

This note will walk you through the process of creating new MXNet operations (or layers).

We try to do our best to provide high speed operators for most common use cases. However, if you do find yourself in need of custom layers, like a novel loss for your research, you have two options:

* Implement an operator in native language (NumpyOp in Python). This is quick to develop but may cause performance issues as it involves copying data and moving it to front-end. We recommend taking this approach first and moving to C++/Cuda if it becomes a bottleneck.

* Implement an operator in C++ and mshadow (and Cuda if necessary). This can be difficult if you are not familiar with MXNet, mashadow or Cuda, but it will give you the best performance. We recommend this approach for performance critical operators.

## Implement Operators in Python
Implementing an operator in Python is similar to creating one in C++ but simpler. Let's create a softmax operator for example. We start by subclassing `mxnet.operator.NumpyOp` and then override a few methods.

First we call our base constructor with `need_top_grad=False`:
```python
class NumpySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(NumpySoftmax, self).__init__(False)
```
This tells the engine that we don't need gradient from layers above for backprop because we are loss layer.

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
        return [data_shape, label_shape], [output_shape]
```
The first dim of an input/output tensor is always batch size. Our label is a set of integers, one for each data entry, and our output has the same shape as input. Infer_shape should always return two lists, even if one of them is empty.

Finally we have finished the preparation and ready to do the real thing:
```python
    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        l = l.reshape((l.size,)).astype(np.int)
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = y
        dx[np.arange(l.shape[0]), l] -= 1.0
```
Remember when you assigning to a tensor, use `x[:] = ...` so that you write to the original array instead of creating a new one.

To use your custom operator, simply create a instance and call it:
```python
mysoftmax = NumpySoftmax()
mlp = mysoftmax(data=fc3, name = 'softmax')
```
Note that you should create a new instance for each symbol.

The complete code for this example can be found at `examples/numpy-ops/numpy_softmax.py`

## Implement Operators in C++
Please refer to [Developer Guide - Operators](https://mxnet.readthedocs.org/en/latest/developer-guide/operator.html) for detail.
