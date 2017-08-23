# Gluon - Neural network building blocks

Gluon package is a high-level interface for MXNet designed to be easy to use while
keeping most of the flexibility of low level API. Gluon supports both imperative
and symbolic programming, making it easy to train complex models imperatively
in Python and then deploy with symbolic graph in C++ and Scala.


```python
# import dependencies
from __future__ import print_function
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd
```

Neural networks (and other machine learning models) can be defined and trained
with `gluon.nn` and `gluon.rnn` package. A typical training script has the following
steps:

- Define network
- Initialize parameters
- Loop over inputs
- Forward input through network to get output
- Compute loss with output and label
- Backprop gradient
- Update parameters with gradient descent.


## Define Network

`gluon.Block` is the basic building block of models. You can define networks by
composing and inheriting `Block`:

```python
class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(6, kernel_size=5)
            self.pool1 = nn.MaxPool2D(pool_size=(2,2))
            self.conv2 = nn.Conv2D(16, kernel_size=5)
            self.pool2 = nn.MaxPool2D(pool_size=(2,2))
            self.fc1 = nn.Dense(120)
            self.fc2 = nn.Dense(84)
            self.fc3 = nn.Dense(10)

    def forward(self, x):
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

## Initialize Parameters

A network must be created and initialized before it can be used:

```python
net = Net()
# Initialize on CPU. Replace with `mx.gpu(0)`, or `[mx.gpu(0), mx.gpu(1)]`,
# etc to use one or more GPUs.
net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
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

## Loss Functions

Loss functions take (output, label) pairs and compute a scalar loss for each sample
in the mini-batch. The scalars measure how far each output is from the label.

There are many predefined loss functions in `gluon.loss`. Here we use
`softmax_cross_entropy_loss` for digit classification.

To compute loss and backprop for one iteration, we do:

```python
label = mx.nd.arange(10)  # dummy label
with autograd.record():
    output = net(data)
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = L(output, label)
    loss.backward()
print('loss:', loss)
print('grad:', net.fc1.weight.grad())
```

## Updating the weights

Now that gradient is computed, we just need to update the weights. This is usually
done with formulas like `weight = weight - learning_rate * grad / batch_size`.
Note we divide gradient by batch_size because gradient is aggregated over the
entire batch. For example,

```python
lr = 0.01
for p in net.collect_params().values():
    p.data()[:] -= lr / data.shape[0] * p.grad()
```

But sometimes you want more fancy updating rules like momentum and Adam, and since
this is a commonly used functionality, gluon provide a `Trainer` class for it:

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

with autograd.record():
    output = net(data)
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = L(output, label)
    loss.backward()

# do the update. Trainer needs to know the batch size of data to normalize
# the gradient by 1/batch_size.
trainer.step(data.shape[0])
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
