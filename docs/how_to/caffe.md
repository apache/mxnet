# How to use MXNet as a (almost) full function Caffe front-end

This tutorial demonstrates how to use MXNet as front-end to Caffe's operators:

* 1) Compile MXNet with Caffe support.

* 2) Embed Caffe's neural network layers into MXNet's symbolic graph.


## Build Caffe
* ~~Download DMLC's forked Caffe [DMLC/Caffe](http://github.com/dmlc/caffe).~~
* Download Caffe repo [BVLC/Caffe](https://github.com/BVLC/caffe), apply patch for mxnet (TODO: add patch link) and compile.

## Compile MXNet with Caffe plugin
* In mxnet folder, open `config.mk` (if you haven't already, copy `make/config.mk` (Linux) or `make/osx.mk` (Mac) into MXNet root folder as `config.mk`) and uncomment the lines `CAFFE_PATH = $(HOME)/caffe` and `MXNET_PLUGINS += plugin/caffe/caffe.mk`. Modify CAFFE_PATH to your caffe package if necessary. 
* Run `make clean && make` to build with caffe support.

## Caffe Operators(Layers)
Caffe's neural network layers are supported by MXNet through `mxnet.symbol.CaffeOperator` symbol.
For example, the following code shows multi-layer perception network and lenet for classifying MNIST digits ([full code](https://github.com/HrWangChengdu/mxnet/blob/master/example/caffe/caffe_net.py)):
```Python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.CaffeOperator(data_0=data, in_num=1, name='fc1', prototxt="layer{ inner_product_param{num_output: 128} }", op_type_string="InnerProduct")
act1 = mx.symbol.CaffeOperator(data_0=fc1, op_type_string="Tanh")
fc2  = mx.symbol.CaffeOperator(data_0=act1, name='fc2', prototxt="layer{ inner_product_param{num_output: 64} }", op_type_string="InnerProduct")
act2 = mx.symbol.CaffeOperator(data_0=fc2, op_type_string="Tanh")
fc3 = mx.symbol.CaffeOperator(data_0=act2, name='fc3', prototxt="layer{ inner_product_param{num_output: 10}}", op_type_string="InnerProduct")
```
Let's break it down. First `data = mx.symbol.Variable('data')` defines a Variable as placeholder for input.
Then it's fed through Caffe's operators with `fc1  = mx.symbol.CaffeOperator(data_0=data, in_num=1, name='fc1', prototxt="layer{ inner_product_param{num_output: 128} }", op_type_string="InnerProduct")`.

Argument `prototxt` is the configuration string for caffe layer. `in_num` specifies number of input and its default value is 1.

`op_type_string` specifies layer type. You can find type-string for rest layers in [link](https://github.com/HrWangChengdu/mxnet/blob/master/plugin/caffe/caffe_operator.cc). Currently, we don't support caffe's data and loss layer.

## Add customized operators
Follow steps below to add new or customized caffe operators:

* 1) Add the layer class into `mxnet/plugin/caffe/caffe_operator.cc` through macro `MXNET_REGISTER_PLUGIN_CAFFE_INIT`.  with header file of class. 

* 2) If new layer contains blobs (weights), add the blob number in function `ListArguments()` of `mxnet/plugin/caffe/caffe_operator-inl.h` .
