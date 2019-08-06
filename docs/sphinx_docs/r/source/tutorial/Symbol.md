# Symbol Interface

Recall that ``NDArray`` is the basic data structure for manipulating numerical values in the **mxnet** package. And just using NDArrays, we can execute a wide range of mathematical operations. So you might wonder, why don’t we just use ``NDArray`` for all computation?

MXNet also provides ``Symbol`` objects, which offer an interface for symbolic programming. With symbolic programming, rather than executing operations step by step, we first define a computation graph. This graph contains placeholders for inputs and designated outputs. We can then compile the graph, yielding a function that can be bound to NDArrays and run. MXNet’s usage of ``Symbol`` is similar to the network configurations used by **Caffe** and the symbolic programming in **Theano**. 

One advantage conferred by the symbolic approach is that **mxnet** can automatically optimize our functions before using them. For example, when we execute mathematical computations in imperative fashion, we don’t know at the time that we run each operation, which values will be needed later on. But with symbolic programming, we declare the required outputs in advance. This means that we can recycle memory allocated in intermediate steps, by performing operations in place.
Note that this tutorial is not meant as a thorough discussion on the comparative strengths of imperative and symbolic programing. We encourage you to read [Symbolic Configuration and Execution in Pictures](https://mxnet.incubator.apache.org/api/python/symbol_in_pictures/symbol_in_pictures.html) (for the Python version of MXNet) in order to gain a detailed understanding of these concepts.

Here, we’ll focus on learning how to use ``Symbol`` functions in the **mxnet** package to define your own custom neural network models in R. In **mxnet**, we can compose Symbols from other Symbols, using operators, such as simple matrix operations (e.g. “+”), or whole neural network layers (e.g. convolution layer). An operator can take multiple input variables, produce multiple output symbols, and maintain internal state symbols.




## Basic Composition of Symbols

The following code creates a two-layer fully-connected feedforward network that outputs a 64-dimensional probability vector (i.e. multilayer perceptron):

```{.python .input  n=1}
require(mxnet)
net <- mx.symbol.Variable("data")
net <- mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=128)
net <- mx.symbol.Activation(data=net, name="relu1", act_type="relu")
net <- mx.symbol.FullyConnected(data=net, name="fc2", num_hidden=64)
net <- mx.symbol.Softmax(data=net, name="out")
class(net)
```

Each symbol takes a (unique) string name (e.g. "data", "fc1", "fc2", "out" in the above example). 
``Variable`` often defines the inputs, or free variables. 
Other symbols take a symbol as the input (``data`` argument), and may accept other hyper parameters, such as the number of hidden neurons (``num_hidden`` argument) or the activation type (``act_type`` argument).

A symbol can be viewed as a function that takes several arguments, whose names are automatically generated and can be retrieved with the following command:

```{.python .input  n=2}
arguments(net)
```

The arguments are the parameters need by each symbol:

- ``data``: Input data needed by the variable "data"
- ``fc1_weight`` and ``fc1_bias``: The weight and bias for the first fully connected layer, "fc1"
- ``fc2_weight`` and ``fc2_bias``: The weight and bias for the second fully connected layer, "fc2"
- ``out_label``: The label needed by the loss

We can also specify the automatically generated names explicitly in a manner that is different from name of the corresponding R variable:

```{.python .input  n=3}
data <- mx.symbol.Variable("data")
w <- mx.symbol.Variable("myweight")
net <- mx.symbol.FullyConnected(data=data, weight=w, name="fc1", num_hidden=128)
arguments(net)
```

## More Complicated Composition of Symbols

MXNet provides well-optimized symbols for commonly used layers in deep learning. 
You can also define new operators in R. The following example first performs an element-wise add between two symbols, then feeds them to the fully connected operator:

```{.python .input  n=4}
lhs <- mx.symbol.Variable("data1")
rhs <- mx.symbol.Variable("data2")
net <- mx.symbol.FullyConnected(data=lhs + rhs, name="fc1", num_hidden=128)
arguments(net)
```

We can construct a symbol more flexibly than by using the single forward composition, for example:

```{.python .input  n=5}
net <- mx.symbol.Variable("data")
net <- mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=128)
net2 <- mx.symbol.Variable("data2")
net2 <- mx.symbol.FullyConnected(data=net2, name="net2", num_hidden=128)
composed.net <- mx.apply(net, data=net2, name="compose")
arguments(composed.net)
```

Above, ``net`` is used as a function to apply to an existing symbol net. The resulting ``composed.net`` will replace the original argument data with ``net2`` instead. 

By defining our own operations on symbols, we can create custom non-standard neural network architectures.
After we have specified a network architecture in terms of symbolic operations, we can subsequently initialize and train the neural network (assigning actual numerical values to these Symbols) via ``mx.model.FeedForward.create`` (as done in the other MXNet tutorials for R).


## How Efficient Is the Symbolic API?

The Symbolic API brings the efficient C++ operations in powerful toolkits, such as CXXNet and Caffe, together with the flexible dynamic NDArray operations. All of the memory and computation resources are allocated statically during ``bind`` operations, to maximize runtime performance and memory utilization.

The coarse-grained operators are equivalent to CXXNet layers, which are extremely efficient. We also provide fine-grained operators for more flexible composition. Because MXNet does more in-place memory allocation, it can be more memory efficient than CXXNet and gets to the same runtime with greater flexibility.
