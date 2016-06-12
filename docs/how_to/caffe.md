# How to use MXNet as a (almost) full function Caffe front-end

This tutorial demonstrates how to use MXNet as front-end to Caffe's operators:

* 1) Compile MXNet with Caffe support.

* 2) Embed Caffe's neural network layers into MXNet's symbolic graph.

## Compile with Caffe
* First download DMLC's forked Caffe [DMLC/Caffe](http://github.com/dmlc/caffe).
* Then, in `config.mk` (if you haven't already, copy `make/config.mk` (Linux) or `make/osx.mk` (Mac) into MXNet root folder as `config.mk`) uncomment the lines `CAFFE_PATH = $(HOME)/caffe` and `MXNET_PLUGINS += plugin/caffe/caffe.mk. Modify CAFFE_PATH to your caffe package if necessary. 
* Run `make clean && make` to build with caffe support.

## Caffe Operators(Layers)
Caffe's neural network layers are supported by MXNet through `mxnet.symbol.CaffeOperator` symbol.
For example, the following code defines a 3 layer DNN for classifying MNIST digits ([full code](https://github.com/HrWangChengdu/mxnet/blob/master/example/image-classification/train_mnist_caffe.py)):
```Python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.CaffeOperator(data = data, name='fc1', para="layer{ inner_product_param{num_output: 128}}", op_type_name="fullyconnected")
act1 = mx.symbol.CaffeOperator(data = fc1, name='act1', para="layer{}", op_type_name="relu")
fc2  = mx.symbol.CaffeOperator(data = act1, name='fc2', para="layer{ inner_product_param{num_output: 64}}", op_type_name="fullyconnected")
act2 = mx.symbol.CaffeOperator(data = fc2, name='act2', para="layer{}", op_type_name="relu")
fc3  = mx.symbol.CaffeOperator(data = act2, name='fc3', para="layer{ inner_product_param{num_output: 10}}", op_type_name="fullyconnected")
mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
```
Let's break it down. First `data = mx.symbol.Variable('data')` defines a Variable as placeholder for input.
Then it's fed through Caffe's operators with `fc1  = mx.symbol.CaffeOperator(data = data, name='fc1', para="layer{ inner_product_param{num_output: 128}}", op_type_name="fullyconnected")
`.

`para` is the prototxt(configuration string) and used to initialize Caffe::LayerParameter. For Caffe's built-in layers, the symbol definition simply calls `caffe::layer_name<float_type>(layerparameter)`.

## Add customized operators
Follow steps below to add new or customized caffe operators:

* 1) Add new enum element to `CaffeEnum::CaffeOpType` in file `caffe_operatoer-inl.h`.

* 2) Then add the layer generation function into `caffe_operator.cc` through macro `DEFINE_CAFFE_LAYER_FN`, also complement the header file of new layer if necessary.

* 3) Under the same file, add key-value pairs <op_name_string, gen_func> & <op_name_string, enum_ele> in function `CaffeTypeNameMap::DoInit()`. The `op_name_string` corresponds to argument `op_type_name` in symbol initialization.
