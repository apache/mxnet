# MXNet

[![Build Status](https://travis-ci.org/dmlc/MXNet.jl.svg?branch=master)](https://travis-ci.org/dmlc/MXNet.jl)
[![Windows Build](https://ci.appveyor.com/api/projects/status/re90njols2th2ide?svg=true)](https://ci.appveyor.com/project/pluskid/mxnet-jl)
[![codecov.io](https://codecov.io/github/dmlc/MXNet.jl/coverage.svg?branch=master)](https://codecov.io/github/dmlc/MXNet.jl?branch=master)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://dmlc.github.io/MXNet.jl/latest)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://dmlc.github.io/MXNet.jl/stable)
[![MXNet](http://pkg.julialang.org/badges/MXNet_0.6.svg)](http://pkg.julialang.org/?pkg=MXNet)
[![License](http://dmlc.github.io/img/apache2.svg)](LICENSE.md)
[![Join the chat at https://gitter.im/dmlc/mxnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/mxnet)


MXNet.jl is the [dmlc/mxnet](https://github.com/dmlc/mxnet) [Julia](http://julialang.org/) package. MXNet.jl brings flexible and efficient GPU computing and state-of-art deep learning to Julia. Some highlight of its features include:

* Efficient tensor/matrix computation across multiple devices, including multiple CPUs, GPUs and distributed server nodes.
* Flexible symbolic manipulation to composite and construct state-of-the-art deep learning models.

Here is an example of how training a simple 3-layer MLP on MNIST looks like:

```julia
using MXNet

mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup model
model = mx.FeedForward(mlp, context=mx.cpu())

# optimization algorithm
optimizer = mx.SGD(lr=0.1, momentum=0.9)

# fit parameters
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)
```

You can also predict using the `model` in the following way:

```julia
probs = mx.predict(model, eval_provider)

# collect all labels from eval data
labels = Array[]
for batch in eval_provider
    push!(labels, copy(mx.get(eval_provider, batch, :softmax_label)))
end
labels = cat(1, labels...)

# Now we use compute the accuracy
correct = 0
for i = 1:length(labels)
    # labels are 0...9
    if indmax(probs[:,i]) == labels[i]+1
        correct += 1
    end
end
accuracy = 100correct/length(labels)
println(mx.format("Accuracy on eval set: {1:.2f}%", accuracy))
```

For more details, please refer to the [documentation](https://dmlc.github.io/MXNet.jl/latest) and [examples](examples).
