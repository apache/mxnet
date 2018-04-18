# How to use Caffe operator in MXNet

[Caffe](http://caffe.berkeleyvision.org/) has been a well-known and widely-used deep learning framework. Now MXNet has supported calling most caffe operators(layers) and loss functions directly in its symbolic graph! Using one's own customized caffe layer is also effortless.

Besides Caffe, MXNet has already embedded Torch modules and its tensor mathematical functions. ([link](https://github.com/dmlc/mxnet/blob/master/docs/faq/torch.md))

This blog demonstrates two steps to use Caffe op in MXNet:

* How to install MXNet with Caffe support.

* How to embed Caffe op into MXNet's symbolic graph.

## Install Caffe With MXNet interface
* Download offical Caffe repository [BVLC/Caffe](https://github.com/BVLC/caffe).
* Download [caffe patch for mxnet interface] (https://github.com/BVLC/caffe/pull/4527.patch). Move patch file under your caffe root folder and apply the patch by `git apply patch_file_name`.
* Install caffe following [official guide](http://caffe.berkeleyvision.org/installation.html).

## Compile with Caffe
* In mxnet folder, open `config.mk` (if you haven't already, copy `make/config.mk` (Linux) or `make/osx.mk` (Mac) into MXNet root folder as `config.mk`) and uncomment the lines `CAFFE_PATH = $(HOME)/caffe` and `MXNET_PLUGINS += plugin/caffe/caffe.mk`. Modify `CAFFE_PATH` to your caffe installation if necessary. 
* Run `make clean && make` to build with caffe support.

## Caffe Operator (Layer)
Caffe's neural network operator and loss functions are supported by MXNet through `mxnet.symbol.CaffeOp` and `mxnet.symbol.CaffeLoss` respectively.
For example, the following code shows multi-layer perception network for classifying MNIST digits ([full code](https://github.com/dmlc/mxnet/blob/master/example/caffe/caffe_net.py)):

### Python
```Python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")
act1 = mx.symbol.CaffeOp(data_0=fc1, prototxt="layer{type:\"TanH\"}")
fc2  = mx.symbol.CaffeOp(data_0=act1, num_weight=2, name='fc2', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 64} }")
act2 = mx.symbol.CaffeOp(data_0=fc2, prototxt="layer{type:\"TanH\"}")
fc3 = mx.symbol.CaffeOp(data_0=act2, num_weight=2, name='fc3', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 10}}")
mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
```

Let's break it down. First `data = mx.symbol.Variable('data')` defines a variable as placeholder for input.
Then it's fed through Caffe operators with `fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")`.

The inputs to caffe op are named as data_i for i=0 ... num_data-1 as `num_data` is the number of inputs. You may skip the argument, as the example does, if its value is 1. While `num_weight` is number of `blobs_`(weights). Its default value is 0, as many ops maintain no weight. `prototxt` is the configuration string.

We could also replace the last line by:

```Python
label = mx.symbol.Variable('softmax_label')
mlp = mx.symbol.CaffeLoss(data=fc3, label=label, grad_scale=1, name='softmax', prototxt="layer{type:\"SoftmaxWithLoss\"}")
```

to use loss function in caffe.
