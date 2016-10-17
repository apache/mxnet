# Symbol and Automatic Differentiation

WIth the computational unit `NDArray`, we need a way to construct neural networks. MXNet provides a symbolic interface named Symbol to do so. The symbol combines both flexibility and efficiency.

## Basic Composition of Symbols

The following codes create a two layer perceptrons network:


```r
require(mxnet)
net <- mx.symbol.Variable("data")
net <- mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=128)
net <- mx.symbol.Activation(data=net, name="relu1", act_type="relu")
net <- mx.symbol.FullyConnected(data=net, name="fc2", num_hidden=64)
net <- mx.symbol.Softmax(data=net, name="out")
class(net)
```

```
## [1] "Rcpp_MXSymbol"
## attr(,"package")
## [1] "mxnet"
```

Each symbol takes a (unique) string name. *Variable* often defines the inputs,
or free variables. Other symbols take a symbol as the input (*data*),
and may accept other hyper-parameters such as the number of hidden neurons (*num_hidden*)
or the activation type (*act_type*).

The symbol can be simply viewed as a function taking several arguments, whose
names are automatically generated and can be get by


```r
arguments(net)
```

```
## [1] "data"       "fc1_weight" "fc1_bias"   "fc2_weight" "fc2_bias"
## [6] "out_label"
```

As can be seen, these arguments are the parameters need by each symbol:

- *data* : input data needed by the variable *data*
- *fc1_weight* and *fc1_bias* : the weight and bias for the first fully connected layer *fc1*
- *fc2_weight* and *fc2_bias* : the weight and bias for the second fully connected layer *fc2*
- *out_label* : the label needed by the loss

We can also specify the automatic generated names explicitly:


```r
data <- mx.symbol.Variable("data")
w <- mx.symbol.Variable("myweight")
net <- mx.symbol.FullyConnected(data=data, weight=w, name="fc1", num_hidden=128)
arguments(net)
```

```
## [1] "data"     "myweight" "fc1_bias"
```

## More Complicated Composition

MXNet provides well-optimized symbols for
commonly used layers in deep learning. We can also easily define new operators
in python.  The following example first performs an elementwise add between two
symbols, then feed them to the fully connected operator.


```r
lhs <- mx.symbol.Variable("data1")
rhs <- mx.symbol.Variable("data2")
net <- mx.symbol.FullyConnected(data=lhs + rhs, name="fc1", num_hidden=128)
arguments(net)
```

```
## [1] "data1"      "data2"      "fc1_weight" "fc1_bias"
```

We can also construct symbol in a more flexible way rather than the single
forward composition we addressed before.


```r
net <- mx.symbol.Variable("data")
net <- mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=128)
net2 <- mx.symbol.Variable("data2")
net2 <- mx.symbol.FullyConnected(data=net2, name="net2", num_hidden=128)
composed.net <- mx.apply(net, data=net2, name="compose")
arguments(composed.net)
```

```
## [1] "data2"       "net2_weight" "net2_bias"   "fc1_weight"  "fc1_bias"
```

In the above example, *net* is used a function to apply to an existing symbol
*net*, the resulting *composed.net* will replace the original argument *data* by
*net2* instead.

## Training a Neural Net.

The [model API](../../../R-package/R/model.R) is a thin wrapper around the symbolic executors to support neural net training.

You are also highly encouraged to read [Symbolic Configuration and Execution in Pictures for python package](../python/symbol_in_pictures.md),
which provides a detailed explanation of concepts in pictures.

## How Efficient is Symbolic API

In short, they design to be very efficient in both memory and runtime.

The major reason for us to introduce Symbolic API, is to bring the efficient C++
operations in powerful tool-kits such as CXXNet and Caffe together with the
flexible dynamic NArray operations. All the memory and computation resources are
allocated statically during Bind, to maximize the runtime performance and memory
utilization.

The coarse grained operators are equivalent to 	CXXNet layers, which are
extremely efficient.  We also provide fine grained operators for more flexible
composition. Because we are also doing more in-place memory allocation, MXNet can
be ***more memory efficient*** than CXXNet, and gets to same runtime, with
greater flexibility.

# Recommended Next Steps
* [Write and use callback functions](http://mxnet.io/tutorials/r/CallbackFunctionTutorial.html)
* [Neural Networks with MXNet in Five Minutes](http://mxnet.io/tutorials/r/fiveMinutesNeuralNetwork.html)
* [Classify Real-World Images with Pre-trained Model](http://mxnet.io/tutorials/r/classifyRealImageWithPretrainedModel.html)
* [Handwritten Digits Classification Competition](http://mxnet.io/tutorials/r/mnistCompetition.html)
* [Character Language Model using RNN](http://mxnet.io/tutorials/r/charRnnModel.html)