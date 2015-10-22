# MXNet

[![Build Status](https://travis-ci.org/dmlc/MXNet.jl.svg?branch=master)](https://travis-ci.org/dmlc/MXNet.jl)
[![Documentation Status](https://readthedocs.org/projects/mxnetjl/badge/?version=latest)](http://mxnetjl.readthedocs.org/en/latest/?badge=latest)
[![License](http://dmlc.github.io/img/apache2.svg)](LICENSE.md)


Julia wrapper of [MXNet](https://github.com/dmlc/mxnet).

```julia
mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.Softmax(name=:softmax)
```
