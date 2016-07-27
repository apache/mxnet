# How to use Caffe Op(Layer) in MXNet

This tutorial demonstrates how to call Caffe operator in MXNet:

* 1) Compile MXNet with Caffe support.

* 2) Embed Caffe's neural network layers into MXNet's symbolic graph.

## Install Caffe With MXNet interface
* Download offical Caffe repository [BVLC/Caffe](https://github.com/BVLC/caffe).
* Download mxnet-interface [patch] (https://github.com/BVLC/caffe/pull/4527.patch). Move patch file under your caffe folder and apply the patch by `git apply 4527.patch`.
* Install caffe following [official guide](http://caffe.berkeleyvision.org/installation.html).

## Compile with Caffe
* In mxnet folder, open `config.mk` (if you haven't already, copy `make/config.mk` (Linux) or `make/osx.mk` (Mac) into MXNet root folder as `config.mk`) and uncomment the lines `CAFFE_PATH = $(HOME)/caffe` and `MXNET_PLUGINS += plugin/caffe/caffe.mk`. Modify `CAFFE_PATH` to your caffe installation if necessary. 
* Run `make clean && make` to build with caffe support.

## Caffe Operators(Layers)
Caffe's neural network layers are supported by MXNet through `mxnet.symbol.CaffeOperator` symbol.
For example, the following code shows multi-layer perception network and lenet for classifying MNIST digits ([full code](https://github.com/HrWangChengdu/mxnet/blob/master/example/caffe/caffe_net.py)):
```Python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")
act1 = mx.symbol.CaffeOp(data_0=fc1, prototxt="layer{type:\"TanH\"}")
fc2  = mx.symbol.CaffeOp(data_0=act1, num_weight=2, name='fc2', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 64} }")
act2 = mx.symbol.CaffeOp(data_0=fc2, prototxt="layer{type:\"TanH\"}")
fc3 = mx.symbol.CaffeOp(data_0=act2, num_weight=2, name='fc3', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 10}}")
mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
```
Let's break it down. First `data = mx.symbol.Variable('data')` defines a Variable as placeholder for input.
Then it's fed through Caffe's operators with `fc1  = mx.symbol.CaffeOperator(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")`.

The inputs to caffe layer are named as data_i for i=0 ... num_data-1 as `num_data` is the number of inputs. You may skip the argument, as the example does, if its value is 1. `num_weight` is number of `blobs_`(weights) in caffe layer. The default value is 0, as most layers, e.g. tanh, owns no weight. `prototxt` is the caffe's layer configuration string. 

We could also replace the last line by:
```Python
label = mx.symbol.Variable('softmax_label')
mlp = mx.symbol.CaffeLoss(data=fc3, label=label, grad_scale=1, name='softmax', prototxt="layer{type:\"SoftmaxWithLoss\"}")
```
to use loss funciton in caffe.

## Use your own customized layers
Running new caffe layer from mxnet is no difference than using regular caffe layers, through rules above. There's no need to add any code in mxnet.
