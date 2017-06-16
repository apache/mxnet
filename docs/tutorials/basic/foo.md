# Foo - High-level Interface

Foo package is a high-level interface for MXNet designed to be easy to use while
keeping most of the flexibility of low level API. Foo supports both imperative
and symbolic programming, making it easy to train complex models imperatively
in Python and then deploy with symbolic graph in C++ and Scala.

This tutorial covers four topics:
- MXNet NDArray as a replacement of numpy for asynchronous scientific computing
across CPU and GPU.
- Automatic differentiation with NDArray.
- Define and train neural network models with Foo's imperative API.
- [TODO] Save trained models as symbolic graph for easy production deployment.

## Setup
First, let's import MXNet and Foo:

```python
from __future__ import print_function
import numpy as np
import mxnet as mx
```

## NDArray

### Creating NDArray

NDArray is similar to numpy's ndarray, but supports asynchronous operations
and GPU. There are many ways to create NDArray.

Construct from (nested) list:
```python
x = mx.nd.array([[1, 2, 3], [4, 5, 6]])
print(x)
```

Construct from numpy array:
```python
x_numpy = np.ones((2, 3))
x = mx.nd.array(x_numpy)
print(x)
```

Array construction routines:
```python
# create an 2x3 array of ones
x = mx.nd.ones((2, 3))
print(x)
# create an 2x3 array of zeros
x = mx.nd.zeros((2, 3))
print(x)
# create an 1d-array of 0 to 5 and reshape to 2x3
x = mx.nd.arange(6).reshape((2, 3))
print(x)
```

You can convert any NDArray to numpy array with `.asnumpy()`:
```python
z = x.asnumpy()
print(z)
```

### NDArray Operations

NDArray supports a wide range of operations. Simple operations can be called
with python syntax:

```python
x = mx.nd.array([[1, 2], [3, 4]])
y = mx.nd.array([[4, 3], [2, 1]])
print(x + y)
```

You can also call operators from the `mxnet.ndarray` (or `mx.nd` for short) name space:

```python
z = mx.nd.add(x, y)
print(z)
```

You can also pass additional flags to operators:

```python
z = mx.nd.sum(x, axis=0)
print('axis=0:', z)
z = mx.nd.sum(x, axis=1)
print('axis=1:', z)
```

By default operators create new NDArrays for return value. You can specify `out`
to use a pre-allocated buffer:

```python
z = mx.nd.empty((2, 2))
mx.nd.add(x, y, out=z)
print(x)
```

### Using GPU

Each NDArray lives on a `Context`. MXNet supports `mx.cpu()` for CPU and `mx.gpu(0)`,
`mx.gpu(1)`, etc for GPU. You can specify context when creating NDArray:

```python
# creates on CPU (the default).
# Replace mx.cpu() with mx.gpu(0) if you have a GPU.
x = mx.nd.zeros((2, 2), ctx=mx.cpu())
print(x)
x = mx.nd.array([[1, 2], [3, 4]], ctx=mx.cpu())
print(x)
```

You can copy arrays between devices with `.copyto()`:

```python
# Copy x to cpu. Replace with mx.gpu(0) if you have GPU.
y = x.copyto(mx.cpu())
# Copy x to another NDArray, possibly on another Context.
x.copyto(y)
print(y)
```

See the [NDArray tutorial](ndarray.md) for a more detailed introduction to
NDArray API.

## Automatic Differentiation

MXNet supports automatic differentiation with the `autograd` package.
`autograd` allows you to differentiate a network of NDArray operations.
This is call define-by-run, i.e., the network is defined on-the-fly by
running forward computation. You can define exotic network structures
and differentiate them, and each iteration can have a totally different
network structure.

```python
form mxnet import autograd
from mxnet.autograd import train_section
```

To use `autograd`, we must first mark variables that require gradient and
attach gradient buffers to them:

```python
x = mx.nd.array([[1, 2], [3, 4]])
dx = mx.nd.zeros_like(x)
x.attach_grad(dx)
```

Now we can define the network while running forward computation by wrapping
it inside a `train_section` (operations out of `train_section` does not define
a graph and cannot be differentiated):

```python
with train_section():
  y = x * 2
  z = y * x
```

Let's backprop with `z.backward()`, which is equivalent to
`z.backward(mx.nd.ones_like(z))`. When z has more than one entry, `z.backward()`
is equivalent to `mx.nd.sum(z).backward()`:

```python
z.backward()
print(x.grad)
```

## Neural Network and Layers

Neural networks (and other machine learning models) can be defined and trained
with `foo.nn` and `foo.rnn` package. A typical training script has the following
steps:

- Define network
- Initialize parameters
- Loop over inputs
- Forward input through network to get output
- Compute loss with output and label
- Backprop gradient
- Update parameters with gradient descent.


### Define Network

`foo.nn.Layer` is the basic building block of models. You can define networks by
composing and inheriting `Layer`:

```python
import mxnet.foo as foo
from mxnet.foo import nn

class Net(nn.Layer):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope:
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(6, kernel_size=5)
            self.pool1 = nn.Pool2D(kernel_size=2)
            self.conv2 = nn.Conv2D(16, kernel_size=5)
            self.pool2 = nn.Pool2D(kernel_size=2)
            self.fc1 = nn.Dense(120)
            self.fc2 = nn.Dense(84)
            self.fc3 = nn.Dense(10)

    def forward(self, F, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Initialize Parameters

A network must be created and initialized before it can be used:

```python
net = Net()
# Initialize on CPU. Replace with `mx.gpu(0)`, or `[mx.gpu(0), mx.gpu(1)]`,
# etc to use one or more GPUs.
net.all_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
```

Note that because we didn't specify input size to layers in Net's constructor,
the shape of parameters cannot be determined at this point. Actual initialization
is deferred to the first forward pass, i.e. if you access `net.fc1.weight.data()`
now an exception will be raised.

You can actually initialize the weights by running a forward pass:

```python
data = mx.nd.random_normal(shape=(10, 1, 32, 32))  # dummy data
output = net(data)
```

Or you can specify input size when creating layers, i.e. `nn.Dense(84, in_units=120)`
instead of `nn.Dense(84)`.

### Loss Functions

Loss functions take (output, label) pairs and compute a scalar loss for each sample
in the mini-batch. The scalars measure how far each output is from the label.

There are many predefined loss functions in `foo.loss`. Here we use
`softmax_cross_entropy_loss` for digit classification.

To compute loss and backprop for one iteration, we do:

```python
label = mx.nd.arange(10)  # dummy label
with train_section():
    output = net(data)
    loss = foo.loss.softmax_cross_entropy_loss(output, label)
    loss.backward()
print('loss:', loss)
print('grad:', net.fc1.weight.grad())
```

### Updating the weights

Now that gradient is computed, we just need to update the weights. This is usually
done with formulas like `weight = weight - learning_rate * grad / batch_size`.
Note we divide gradient by batch_size because gradient is aggregated over the
entire batch. For example,

```python
lr = 0.01
for p in net.all_params().values():
    p.data()[:] -= lr / data.shape[0] * p.grad()
```

But sometimes you want more fancy updating rules like momentum and Adam, and since
this is a commonly used functionality, foo provide a `Trainer` class for it:

```python
trainer = foo.Trainer(net.all_params(), 'sgd', {'learning_rate': 0.01})

with train_section():
    output = net(data)
    loss = foo.loss.softmax_cross_entropy_loss(output, label)
    loss.backward()

# do the update. Trainer needs to know the batch size of data to normalize
# the gradient by 1/batch_size.
trainer.step(data.shape[0])
```
