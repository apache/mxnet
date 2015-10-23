# MXNet

[![Build Status](https://travis-ci.org/dmlc/MXNet.jl.svg?branch=master)](https://travis-ci.org/dmlc/MXNet.jl)
[![Documentation Status](https://readthedocs.org/projects/mxnetjl/badge/?version=latest)](http://mxnetjl.readthedocs.org/en/latest/?badge=latest)
[![License](http://dmlc.github.io/img/apache2.svg)](LICENSE.md)


Julia wrapper of [MXNet](https://github.com/dmlc/mxnet).

```julia
using MXNet

mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.Softmax(name=:softmax)

# data provider
batch_size = 100
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup estimator
estimator = mx.FeedForward(mlp, context=mx.cpu())

# optimizer
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

# fit parameters
mx.fit(estimator, optimizer, train_provider, epoch_stop=20, eval_data=eval_provider)
```
