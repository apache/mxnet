# How to Create New Operations (Layers)

This note will walk you through the process of creating new MXNet operations (or layers).

We try to do our best to provide high speed operators for most common use cases. However, if you do find yourself in need of custom layers, like a novel loss for your research, you have two options:

* 1) Use native language and it's matrix library (e.g. numpy in Python). This requires least effort and knowledge of MXNet. But impairs performance as it is CPU based.

* 2) Use native language, mxnet.rtc and mxnet.ndarray. This gives you most of the performance of 3) and most of the convenience of 1), but requires more knowledge of MXNet. You can write CUDA kernels in python and compile with during runtime.

* 3) Use C++/MShadow(CUDA). This can be difficult if you are not familiar with MXNet, mashadow or Cuda, but it will give you the best performance.

## Python/Numpy
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

## Python/NDArray(MXRtc)
Again we use Softmax as an example. We start by subclassing `mxnet.operator.NDArrayOp` and then override a few methods.

First we call our base constructor with `need_top_grad=False`:
class NDArraySoftmax(mx.operator.NDArrayOp):
    def __init__(self):
        super(NDArraySoftmax, self).__init__(False)
        self.fwd_kernel = None
        self.bwd_kernel = None

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
        if self.fwd_kernel is None:
            self.fwd_kernel = mx.rtc('softmax', [('x', x)], [('y', y)], """
int i = threadIdx.x + blockIdx.x*blockDim.x;
float max_x = x[i*x_dims[1]];
for (int j = 1; j < x_dims[1]; ++j) {
    if (max_x < x[i*x_dims[1]+j]) {
        max_x = x[i*x_dims[1]+j];
    }
}
float sum = 0.0f;
for (int j = 0; j < x_dims[1]; ++j) {
    sum += expf(x[i*x_dims[1]+j]-max_x);
}
for (int j = 0; j < x_dims[1]; ++j) {
    y[i*x_dims[1]+j] = expf(x[i*x_dims[1]+j]-max_x)/sum;
}
""")
        self.fwd_kernel.push([x], [y], (1, 1, 1), (x.shape[0], 1, 1))
```
This may seem hard to understand but let's break it up. First in_data and out_data are lists holding ndarrays in the same order as declared in list_arguments and list_outputs. We then construct a mx.rtc object that holds a CUDA kernel. mx.rtc takes 4 arguments: name of the kernel function, list of inputs as (name, ndarray) tuples, list of outputs, and finally the kernel function.

You may have noticed that we only defined the body of the kernel but not the definition. This is because mx.rtc will decorate kernel.
For example, if `name = "mykernel"` and `inputs = [('x', mx.nd.zeros((10,)))]`,  `outputs = [('y', mx.nd.zeros((10,)))]`, `kernel = "y[threadIdx.x] = x[threadIdx.x];"`, the kernel that is compile will be:
```C
extern "C" __global__ mykernel(float *x, float *y) {
    const int x_ndim = 1;
    const int x_dims = { 10 };
    const int y_ndim = 1;
    const int y_dims = { 10 };

    y[threadIdx.x] = x[threadIdx.x];
}
```
Finally, we launch the kernel with `self.fwd_kernel.push([x], [y], (1, 1, 1), (x.shape[0], 1, 1))`, where `(1, 1, 1)` and  `(x.shape[0], 1, 1)` are the grid and block dimensions.

## C++/MShadow(CUDA)
Please refer to [Developer Guide - Operators](https://mxnet.readthedocs.org/en/latest/system/operator.html) for detail.
