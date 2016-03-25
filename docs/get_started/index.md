# Introduction

Let's start with training a simple
[multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
(MLP) on
the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/) to get
the basic idea of how to use MXNet.

## Train MLP on MNIST

On MNIST, each example consists of a 28 x 28 gray image of a handwritten digit
such as
<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mnist.png"
height="36" >
and its label, which is an integer between 0 and 9. Denote by *x* and *y* the
784-length vector of the image pixels and the label, respectively.

MLP with no hidden layer predicts the label probabilities by

```eval_rst
.. math ::
  \textrm{softmax}(W x + b)
```

here *W* is a 784-by-10 weight matrix, *b* is a 10-length bias vector. The
*softmax* normalizes a vector into a probability distribution, namely

```eval_rst
.. math ::
  \textrm{softmax}(x) = \left[ \ldots, \frac{\exp(x_i)}{\sum_j \exp(x_j)}\ldots \right]
```

We can stack the layers one by one to get a MLP with multiple hidden layers. Let
*x<sub>0</sub> = x* be the output of layer 0. Then layer *i*  outputs a
*n<sub>i</sub>*-length vector

```eval_rst
.. math ::
  x_i = \sigma_i (W_i x_{i-1} + b_i)
```

where *σ<sub>i</sub>* is the activation function such as *tanh*, and
*W<sub>i</sub>* is of size *n<sub>i</sub>*-by-*n<sub>i-1</sub>*. Next we apply
*softmax* to the last layer to obtain the prediction.

The goal of training is to obtain both weights and bias for each layer to
minimize the difference between the predicted label and the real label on the
training data.

In the following section we will show how to implement the training program
using different languages in MXNet.

### Python

We first import MXNet
```python
import mxnet as mx
```

Then we declare the data iterators to the training and validation datasets
```python
train = mx.io.MNISTIter(
    image      = "mnist/train-images-idx3-ubyte",
    label      = "mnist/train-labels-idx1-ubyte",
    batch_size = 128,
    data_shape = (784, ))
val   = mx.io.MNISTIter(...)
```

and declare a two-layer MLP
```python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, num_hidden=10)
mlp  = mx.symbol.SoftmaxOutput(data = fc3)
```

Next we train a model on the data
```python
model = mx.model.FeedForward(
    symbol = mlp,
    num_epoch = 20,
    learning_rate = .1)
model.fit(X = train, eval_data = val)
```

Finally we can predict by
```python
test = mx.io.MNISTIter(...)
model.predict(X = test)
```

### R

### Scala

### Julia

We first import MXNet

```julia
using MXNet
```

Then load data
```julia
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size)
```

and define the MLP
```julia
mlp = @mx.chain mx.Variable(:data)  =>
  mx.FullyConnected(num_hidden=128) =>
  mx.Activation(act_type=:relu)     =>
  mx.FullyConnected(num_hidden=64)  =>
  mx.Activation(act_type=:relu)     =>
  mx.FullyConnected(num_hidden=10)  =>
  mx.SoftmaxOutput()
```

The model can be trained by

```julia
model = mx.FeedForward(mlp, context=mx.cpu())
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)
```

and finally predict by
```julia
probs = mx.predict(model, test_provider)
```

## Tensor Computation

Next we briefly introduce the tensor computation interface, which is often more
flexiable to use than the previous symbolic interface. It is often used to
implement the layers, define weight updating rules, and debug.


### Python

The python inferface is similar to `numpy.NDArray`.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3),
... mx.gpu())
>>> print (a * 2).asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

### R

### Scala

### Julia

## Recommended Next Steps

- [Build and Install](build.html)
- [Step by Step Tutorials](../tutorials/index.html)
- [How-Tos](../tutorials/index.html)
