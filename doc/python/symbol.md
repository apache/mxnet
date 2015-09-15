# Symbolic and Automatic Differentiation

Now you have seen the power of NArray of MXNet. It seems to be interesting and
we are ready to build some real deep learning.  Hmm, this seems to be really
exciting, but wait, do we need to build things from scratch?  It seems that we
need to re-implement all the layers in deep learning toolkits such as
[CXXNet](https://github.com/dmlc/cxxnet) in NArray?  Well, you do not have
to. There is a Symbolic API in MXNet that readily helps you to do all these.

More importantly, the Symbolic API is designed to bring in the advantage of C++
static layers(operators) to ***maximumly optimizes the performance and memory***
that is even better than CXXNet. Sounds exciting? Let us get started on this.

## Creating Symbols

A common way to create a neural network is to create it via some way of
configuration file or API.  The following code creates a configuration two layer
perceptrons.

```python
import mxnet.symbol as sym
data = sym.Variable('data')
net = sym.FullyConnected(data=data, name='fc1', num_hidden=128)
net = sym.Activation(data=net, name='relu1', act_type="relu")
net = sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = sym.Softmax(data=net, name = 'sm')
```

If you are familiar with tools such as cxxnet or caffe, the ```Symbol``` object
is like configuration files that configures the network structure. If you are
more familiar with tools like theano, the ```Symbol``` object something that
defines the computation graph. Basically, it creates a computation graph that
defines the forward pass of neural network.

The Configuration API allows you to define the computation graph via
compositions.  If you have not used symbolic configuration tools like theano
before, one thing to note is that the ```net``` can also be viewed as function
that have input arguments.

You can get the list of arguments by calling ```Symbol.list_arguments```.

```python
>>> net.list_arguments()
['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias']
```

In our example, you can find that the arguments contains the parameters in each
layer, as well as input data.  One thing that worth noticing is that the
argument names like ```fc1_weight``` are automatically generated because it was
not specified in creation of fc1.  You can also specify it explicitly, like the
following code.

```python
>>> import mxnet.symbol as sym
>>> data = sym.Variable('data')
>>> w = sym.Variable('myweight')
>>> net = sym.FullyConnected(data=data, weight=w,
                             name='fc1', num_hidden=128)
>>> net.list_arguments()
['data', 'myweight', 'fc1_bias']
```

Besides the coarse grained neuralnet operators such as FullyConnected,
Convolution.  MXNet also provides fine graned operations such as elementwise
add, multiplications.  The following example first performs an elementwise add
between two symbols, then feed them to the FullyConnected operator.

```python
>>> import mxnet.symbol as sym
>>> lhs = sym.Variable('data1')
>>> rhs = sym.Variable('data2')
>>> net = sym.FullyConnected(data=lhs + rhs,
                             name='fc1', num_hidden=128)
>>> net.list_arguments()
['data1', 'data2', 'fc1_weight', 'fc1_bias']
```

## More Complicated Composition

In the previous example, Symbols are constructed in a forward compositional way.
Besides doing things in a forward compistion way. You can also treat composed
symbols as functions, and apply them to existing symbols.

```python
>>> import mxnet.symbol as sym
>>> data = sym.Variable('data')
>>> net = sym.FullyConnected(data=data,
                             name='fc1', num_hidden=128)
>>> net.list_arguments()
['data', 'fc1_weight', 'fc1_bias']
>>> data2 = sym.Variable('data2')
>>> in_net = sym.FullyConnected(data=data,
                                name='in', num_hidden=128)
>>> composed_net = net(data=in_net, name='compose')
>>> composed_net.list_arguments()
['data2', 'in_weight', 'in_bias', 'compose_fc1_weight', 'compose_fc1_bias']
```

In the above example, net is used a function to apply to an existing symbol
```in_net```, the resulting composed_net will replace the original ```data``` by
the the in_net instead. This is useful when you want to change the input of some
neural-net to be other structure.

## Shape Inference

Now we have defined the computation graph. A common problem in the computation
graph, is to figure out shapes of each parameters.  Usually, we want to know the
shape of all the weights, bias and outputs.

You can use ```Symbol.infer_shape``` to do that. THe shape inference function
allows you to pass in shapes of arguments that you know,
and it will try to infer the shapes of all arguments and outputs.

```python
>>> import mxnet.symbol as sym
>>> data = sym.Variable('data')
>>> net = sym.FullyConnected(data=data, name='fc1',
                             num_hidden=10)
>>> arg_shape, out_shape = net.infer_shape(data=(100, 100))
>>> dict(zip(net.list_arguments(), arg_shape))
{'data': (100, 100), 'fc1_weight': (10, 100), 'fc1_bias': (10,)}
>>> out_shape
[(100, 10)]
```

In common practice, you only need to provide the shape of input data, and it
will automatically infers the shape of all the parameters.  You can always also
provide more shape information, such as shape of weights.  The ```infer_shape```
will detect if there is inconsitency in the shapes, and raise an Error if some
of them are inconsistent.

## Bind the Symbols

Symbols are configuration objects that represents a computation graph (a
configuration of neuralnet).  So far we have introduced how to build up the
computation graph (i.e. a configuration).  The remaining question is, how we can
do computation using the defined graph.

TODO.

## How Efficient is Symbolic API

In short, they design to be very efficienct in both memory and runtime.

The major reason for us to introduce Symbolic API, is to bring the efficient C++
operations in powerful toolkits such as cxxnet and caffe together with the
flexible dynamic NArray operations. All the memory and computation resources are
allocated statically during Bind, to maximize the runtime performance and memory
utilization.

The coarse grained operators are equivalent to cxxnet layers, which are
extremely efficient.  We also provide fine grained operators for more flexible
composition. Because we are also doing more inplace memory allocation, mxnet can
be ***more memory efficient*** than cxxnet, and gets to same runtime, with
greater flexiblity.

## Symbol API

```eval_rst
.. automodule:: mxnet.symbol
    :members:
```


## Executor API
```eval_rst
.. automodule:: mxnet.executor
    :members:
```
