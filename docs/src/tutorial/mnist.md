Digit Recognition on MNIST
==========================

In this tutorial, we will work through examples of training a simple
multi-layer perceptron and then a convolutional neural network (the
LeNet architecture) on the [MNIST handwritten digit
dataset](http://yann.lecun.com/exdb/mnist/). The code for this tutorial
could be found in
[examples/mnist](https://github.com/dmlc/MXNet.jl/tree/master/examples/mnist).  There are also two Jupyter notebooks that expand a little more on the [MLP](https://github.com/ultradian/julia_notebooks/blob/master/mxnet/mnistMLP.ipynb) and the [LeNet](https://github.com/ultradian/julia_notebooks/blob/master/mxnet/mnistLenet.ipynb), using the more general `ArrayDataProvider`. 

Simple 3-layer MLP
------------------

This is a tiny 3-layer MLP that could be easily trained on CPU. The
script starts with

```julia
using MXNet
```

to load the `MXNet` module. Then we are ready to define the network
architecture via the [symbolic API](../user-guide/overview.md). We start
with a placeholder `data` symbol,

```julia
data = mx.Variable(:data)
```

and then cascading fully-connected layers and activation functions:

```julia
fc1  = mx.FullyConnected(data, name=:fc1, num_hidden=128)
act1 = mx.Activation(fc1, name=:relu1, act_type=:relu)
fc2  = mx.FullyConnected(act1, name=:fc2, num_hidden=64)
act2 = mx.Activation(fc2, name=:relu2, act_type=:relu)
fc3  = mx.FullyConnected(act2, name=:fc3, num_hidden=10)
```

Note each composition we take the previous symbol as the first argument,
forming a feedforward chain. The architecture looks like

```
Input --> 128 units (ReLU) --> 64 units (ReLU) --> 10 units
```

where the last 10 units correspond to the 10 output classes (digits
0,...,9). We then add a final `SoftmaxOutput` operation to turn the
10-dimensional prediction to proper probability values for the 10
classes:

```julia
mlp  = mx.SoftmaxOutput(fc3, name=:softmax)
```

As we can see, the MLP is just a chain of layers. For this case, we can
also use the `mx.chain` macro. The same architecture above can be
defined as

```julia
mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.SoftmaxOutput(name=:softmax)
```

After defining the architecture, we are ready to load the MNIST data.
MXNet.jl provide built-in data providers for the MNIST dataset, which
could automatically download the dataset into
`Pkg.dir("MXNet")/data/mnist` if necessary. We wrap the code to
construct the data provider into `mnist-data.jl` so that it could be
shared by both the MLP example and the LeNet ConvNets example.

```julia
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size)
```

If you need to write your own data providers for customized data format,
please refer to [`mx.AbstractDataProvider`](@ref).

Given the architecture and data, we can instantiate an *model* to do the
actual training. `mx.FeedForward` is the built-in model that is suitable
for most feed-forward architectures. When constructing the model, we
also specify the *context* on which the computation should be carried
out. Because this is a really tiny MLP, we will just run on a single CPU
device.

```julia
model = mx.FeedForward(mlp, context=mx.cpu())
```

You can use a `mx.gpu()` or if a list of devices (e.g.
`[mx.gpu(0), mx.gpu(1)]`) is provided, data-parallelization will be used
automatically. But for this tiny example, using a GPU device might not
help.

The last thing we need to specify is the optimization algorithm (a.k.a.
*optimizer*) to use. We use the basic SGD with a fixed learning rate 0.1
, momentum 0.9 and weight decay 0.00001:

```julia
optimizer = mx.SGD(η=0.1, μ=0.9, λ=0.00001)
```

Now we can do the training. Here the `n_epoch` parameter specifies that
we want to train for 20 epochs. We also supply a `eval_data` to monitor
validation accuracy on the validation set.

```julia
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)
```

Here is a sample output

```
INFO: Start training on [CPU0]
INFO: Initializing parameters...
INFO: Creating KVStore...
INFO: == Epoch 001 ==========
INFO: ## Training summary
INFO:       :accuracy = 0.7554
INFO:            time = 1.3165 seconds
INFO: ## Validation summary
INFO:       :accuracy = 0.9502
...
INFO: == Epoch 020 ==========
INFO: ## Training summary
INFO:       :accuracy = 0.9949
INFO:            time = 0.9287 seconds
INFO: ## Validation summary
INFO:       :accuracy = 0.9775
```

Convolutional Neural Networks
-----------------------------

In the second example, we show a slightly more complicated architecture
that involves convolution and pooling. This architecture for the MNIST
is usually called the \[LeNet\]\_. The first part of the architecture is
listed below:

```julia
# input
data = mx.Variable(:data)

# first conv
conv1 = @mx.chain mx.Convolution(data, kernel=(5,5), num_filter=20)  =>
                  mx.Activation(act_type=:tanh) =>
                  mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))

# second conv
conv2 = @mx.chain mx.Convolution(conv1, kernel=(5,5), num_filter=50) =>
                  mx.Activation(act_type=:tanh) =>
                  mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))
```

We basically defined two convolution modules. Each convolution module is
actually a chain of `Convolution`, `tanh` activation and then max
`Pooling` operations.

Each sample in the MNIST dataset is a 28x28 single-channel grayscale
image. In the tensor format used by `NDArray`, a batch of 100 samples is
a tensor of shape `(28,28,1,100)`. The convolution and pooling operates
in the spatial axis, so `kernel=(5,5)` indicate a square region of
5-width and 5-height. The rest of the architecture follows as:

```julia
# first fully-connected
fc1   = @mx.chain mx.Flatten(conv2) =>
                  mx.FullyConnected(num_hidden=500) =>
                  mx.Activation(act_type=:tanh)

# second fully-connected
fc2   = mx.FullyConnected(fc1, num_hidden=10)

# softmax loss
lenet = mx.Softmax(fc2, name=:softmax)
```

Note a fully-connected operator expects the input to be a matrix.
However, the results from spatial convolution and pooling are 4D
tensors. So we explicitly used a `Flatten` operator to flat the tensor,
before connecting it to the `FullyConnected` operator.

The rest of the network is the same as the previous MLP example. As
before, we can now load the MNIST dataset:

```julia
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size; flat=false)
```

Note we specified `flat=false` to tell the data provider to provide 4D
tensors instead of 2D matrices because the convolution operators needs
correct spatial shape information. We then construct a feedforward model
on GPU, and train it.

```julia
# fit model
model = mx.FeedForward(lenet, context=mx.gpu())

# optimizer
optimizer = mx.SGD(η=0.05, μ=0.9, λ=0.00001)

# fit parameters
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)
```

And here is a sample of running outputs:

```
INFO: == Epoch 001 ==========
INFO: ## Training summary
INFO:       :accuracy = 0.6750
INFO:            time = 4.9814 seconds
INFO: ## Validation summary
INFO:       :accuracy = 0.9712
...
INFO: == Epoch 020 ==========
INFO: ## Training summary
INFO:       :accuracy = 1.0000
INFO:            time = 4.0086 seconds
INFO: ## Validation summary
INFO:       :accuracy = 0.9915
```

Predicting with a trained model
-------------------------------

Predicting with a trained model is very simple. By calling `mx.predict`
with the model and a data provider, we get the model output as a Julia
Array:

```julia
probs = mx.predict(model, eval_provider)
```

The following code shows a stupid way of getting all the labels from the
data provider, and compute the prediction accuracy manually:

```julia
# collect all labels from eval data
labels = reduce(
  vcat,
  copy(mx.get(eval_provider, batch, :softmax_label)) for batch ∈ eval_provider)
# labels are 0...9
labels .= labels .+ 1

# Now we use compute the accuracy
pred = map(i -> indmax(probs[1:10, i]), 1:size(probs, 2))
correct = sum(pred .== labels)
@printf "Accuracy on eval set: %.2f%%\n" 100correct/length(labels)
```

Alternatively, when the dataset is huge, one can provide a callback to
`mx.predict`, then the callback function will be invoked with the
outputs of each mini-batch. The callback could, for example, write the
data to disk for future inspection. In this case, no value is returned
from `mx.predict`. See also predict.
