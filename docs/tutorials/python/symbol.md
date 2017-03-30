# Symbolic and Automatic Differentiation

NDArray is the basic computation unit in MXNet. MXNet also provides a
symbolic interface, named Symbol, to simplify constructing neural networks. Symbol combines flexibility and efficiency. It is similar to
the network configuration in [Caffe](http://caffe.berkeleyvision.org/) and
[CXXNet](https://github.com/dmlc/cxxnet) and the symbols define
the computation graph as in [Theano](http://deeplearning.net/software/theano/).

## Basic Composition of Symbols

The following code creates a two-layer perceptron network:

 ```python
    >>> import mxnet as mx
    >>> net = mx.symbol.Variable('data')
    >>> net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
    >>> net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    >>> net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=64)
    >>> net = mx.symbol.SoftmaxOutput(data=net, name='out')
    >>> type(net)
    <class 'mxnet.symbol.Symbol'>
 ```

Each symbol takes a (unique) string name. *Variable* often defines the inputs,
or free variables. Other symbols take a symbol as their input (*data*),
and might accept other hyper parameters, such as the number of hidden neurons (*num_hidden*)
or the activation type (*act_type*).

The symbol can be seen simply as a function taking several arguments whose
names are automatically generated and can be got with the following:

 ```python
    >>> net.list_arguments()
    ['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'out_label']
 ```

 These arguments are the parameters needed by each symbol:

- *data*: Input data needed by the variable *data*
- *fc1_weight* and *fc1_bias*: The weight and bias for the first fully connected layer *fc1*
- *fc2_weight* and *fc2_bias*: The weight and bias for the second fully connected layer *fc2*
- *out_label*: The label needed by the loss

We can also specify the automatically generated names explicitly:

 ```python
    >>> net = mx.symbol.Variable('data')
    >>> w = mx.symbol.Variable('myweight')
    >>> net = mx.symbol.FullyConnected(data=net, weight=w, name='fc1', num_hidden=128)
    >>> net.list_arguments()
    ['data', 'myweight', 'fc1_bias']
 ```

## More Complicated Composition

MXNet provides well-optimized symbols for layers
commonly used in deep learning (see
[src/operator](https://github.com/dmlc/mxnet/tree/master/src/operator)). We can also easily define new operators
in Python.  The following example first performs an element-wise add between two
symbols, then feeds them to the fully connected operator:

 ```python
    >>> lhs = mx.symbol.Variable('data1')
    >>> rhs = mx.symbol.Variable('data2')
    >>> net = mx.symbol.FullyConnected(data=lhs + rhs, name='fc1', num_hidden=128)
    >>> net.list_arguments()
    ['data1', 'data2', 'fc1_weight', 'fc1_bias']
 ```

We can also construct a symbol in a more flexible way than the single
forward composition exemplified in the preceding example:

 ```python
    >>> net = mx.symbol.Variable('data')
    >>> net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
    >>> net2 = mx.symbol.Variable('data2')
    >>> net2 = mx.symbol.FullyConnected(data=net2, name='net2', num_hidden=128)
    >>> composed_net = net(data=net2, name='compose')
    >>> composed_net.list_arguments()
    ['data2', 'net2_weight', 'net2_bias', 'fc1_weight', 'fc1_bias']
 ```

In the preceding example, *net* is used as a function to apply to an existing symbol
*net*, and the resulting *composed_net* will replace the original argument *data* with
*net2*.

Once you start building some bigger networks, you might want to name some symbols with a common prefix to outline the structure of your network. You can use the [Prefix](https://github.com/dmlc/mxnet/blob/master/python/mxnet/name.py) NameManager as follow:

```python
   >>> data = mx.sym.Variable("data")
   >>> net = data
   >>> n_layer = 2
   >>> for i in range(n_layer):
   ...     with mx.name.Prefix("layer%d_" % (i + 1)):
   ...         net = mx.sym.FullyConnected(data=net, name="fc", num_hidden=100)
   ...
   >>> net.list_arguments()
   ['data', 'layer1_fc_weight', 'layer1_fc_bias', 'layer2_fc_weight', 'layer2_fc_bias']
```

## Argument Shape Inference

Now we know how to define a symbol. Next, we can infer the shapes of
all of the arguments it needs given the shape of its input data:

 ```python
    >>> net = mx.symbol.Variable('data')
    >>> net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=10)
    >>> arg_shape, out_shape, aux_shape = net.infer_shape(data=(100, 100))
    >>> dict(zip(net.list_arguments(), arg_shape))
    {'data': (100, 100), 'fc1_weight': (10, 100), 'fc1_bias': (10,)}
    >>> out_shape
    [(100, 10)]
 ```

We can use this shape inference as an early debugging mechanism to detect
shape inconsistency.

## Bind the Symbols and Run

Now we can bind the free variables of the symbol and perform forward and backward operations.
The ```bind``` function will create a ```Executor``` that can be used to carry out the real computations:

 ```python
    >>> # define computation graphs
    >>> A = mx.symbol.Variable('A')
    >>> B = mx.symbol.Variable('B')
    >>> C = A * B
    >>> a = mx.nd.ones(3) * 4
    >>> b = mx.nd.ones(3) * 2
    >>> # bind the symbol with real arguments
    >>> c_exec = C.bind(ctx=mx.cpu(), args={'A' : a, 'B': b})
    >>> # do forward pass calculation.
    >>> c_exec.forward()
    >>> c_exec.outputs[0].asnumpy()
    [ 8.  8.  8.]
 ```
For neural nets, a more commonly used pattern is ```simple_bind```, which creates all of the argument arrays for you. Then you can call ```forward```, and ```backward``` (if the gradient is needed)
to get the gradient:

 ```python
    >>> # define computation graphs
    >>> net = some symbol
    >>> texec = net.simple_bind(data=input_shape)
    >>> texec.forward()
    >>> texec.backward()
 ```
The [model API](model.md) is a thin wrapper around the symbolic executors to support neural net training.

We strongly encouraged you to read [Symbolic Configuration and Execution in Pictures](symbol_in_pictures.md),
which provides a detailed explanation of the concepts in pictures.

## How Efficient Is the Symbolic API?

In short, it is designed to be very efficient in both memory and runtime.

The major reason for introducing the Symbolic API is to bring the efficient C++
operations in powerful toolkits, such as CXXNet and Caffe, together with the
flexible dynamic NDArray operations. To maximize runtime performance and memory
utilization, all of the memory and computation resources are
allocated statically during the bind operation.

The coarse-grained operators are equivalent to CXXNet layers, which are
extremely efficient.  We also provide fine-grained operators for more flexible
composition. Because we are also performing more in-place memory allocation, MXNet can
be more memory efficient than CXXNet, and achieves the same runtime, with
greater flexibility.

## Next Steps
* [KVStore](kvstore.md)
