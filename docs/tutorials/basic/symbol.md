# Symbol - Neural network graphs and auto-differentiation

NDArray is the basic computation unit in MXNet. MXNet also provides a symbolic
interface, named Symbol, also known as `mxnet.symbol`, or `mxnet.sym` for short.
Symbol combines flexibility and efficiency and simplifies constructing neural
networks.

It is similar to the network configuration in [Caffe](http://caffe.berkeleyvision.org/)
and [CXXNet](https://github.com/dmlc/cxxnet) and the symbols define the computation
graph as in [Theano](http://deeplearning.net/software/theano/).

A symbol also represents a multi-output symbolic expression and is used to declare
the computation. Symbols are composited by operators, such as simple matrix
operations (e.g. "+"), or a neural network layer (e.g. convolution layer).
An operator can take several input variables, produce more than one output variables
and have internal state variables. A variable can be either free, which we can bind
with value later, or an output of another symbol.

We strongly encouraged you to read
[Symbolic Configuration and Execution in Pictures](symbol_in_pictures.md), which
provides a detailed explanation of the concepts in pictures.

There are a few different ways to compose a `Symbol`.

## Basic Symbol Composition

### Basic Operators

The following example composites a simple expression `a+b`. We first create the
placeholders `a` and `b` with names using `mx.sym.Variable` and then construct
the desired symbol by using the operator `+`. If the string name is not given to
the symbol during creation, MXNet will automatically generate a unique name for
it, which is the case for `c`.

```python
import mxnet as mx
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b
(a, b, c)
```

Most `NDArray` operators can also be applied to `Symbol`, for example:

```python
# elemental wise multiplication
d = a * b
# matrix multiplication
e = mx.sym.dot(a, b)
# reshape
f = mx.sym.reshape(d+e, shape=(1,4))
# broadcast
g = mx.sym.broadcast_to(f, shape=(2,4))
# plot
mx.viz.plot_network(symbol=g)
```

The computations declared in the above examples can be bound to the input data
for evaluation by using `bind` method. It is discussed in the
[symbol manipulation](#Symbol Manipulation) section.

### Basic Neural Networks

Besides the basic operators, `Symbol` has a rich set of neural network layers.
The following example constructs a two layer fully connected neural network
and then visualize the structure of that network given the input data shape.

```python
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(data=net, name='out')
mx.viz.plot_network(net, shape={'data':(100,200)})
```

Each symbol takes a (unique) string name. *Variable* often defines the inputs,
or free variables. Other symbols take a symbol as their input (*data*), and
might accept other hyperparameters, such as the number of hidden neurons
(*num_hidden*) or the activation type (*act_type*).

The symbol can be seen simply as a function taking several arguments whose names
are automatically generated and can be retrieved with the following method call:

```python
net.list_arguments()
```

These arguments are the parameters needed by each symbol:

- *data*: Input data needed by the variable *data*.
- *fc1_weight* and *fc1_bias*: The weight and bias for the first fully connected layer *fc1*.
- *fc2_weight* and *fc2_bias*: The weight and bias for the second fully connected layer *fc2*.
- *out_label*: The label needed by the loss.

We can also specify the automatically generated names explicitly:

```python
net = mx.symbol.Variable('data')
w = mx.symbol.Variable('myweight')
net = mx.symbol.FullyConnected(data=net, weight=w, name='fc1', num_hidden=128)
net.list_arguments()
```

## More Complicated Composition

MXNet provides well-optimized symbols for layers commonly used in deep learning
(see [src/operator](https://github.com/dmlc/mxnet/tree/master/src/operator)).
We can also easily define new operators in Python. The following example first
performs an element-wise add between two symbols, then feeds them to the fully
connected operator:

```python
lhs = mx.symbol.Variable('data1')
rhs = mx.symbol.Variable('data2')
net = mx.symbol.FullyConnected(data=lhs + rhs, name='fc1', num_hidden=128)
net.list_arguments()
```

We can also construct a symbol in a more flexible way than the single forward
composition exemplified in the preceding example:

```python
net = mx.symbol.Variable('data')
net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
net2 = mx.symbol.Variable('data2')
net2 = mx.symbol.FullyConnected(data=net2, name='net2', num_hidden=128)
composed_net = net(data=net2, name='compose')
composed_net.list_arguments()
```

In the preceding example, *net* is used as a function to apply to an existing
symbol *net*, and the resulting *composed_net* will replace the original
argument *data* with *net2*.

Once you start building some bigger networks, you might want to name some
symbols with a common prefix to outline the structure of your network. You can
use the
[Prefix](https://github.com/dmlc/mxnet/blob/master/python/mxnet/name.py)
NameManager as follows:

```python
data = mx.sym.Variable("data")
net = data
n_layer = 2
for i in range(n_layer):
   with mx.name.Prefix("layer%d_" % (i + 1)):
   net = mx.sym.FullyConnected(data=net, name="fc", num_hidden=100)
net.list_arguments()
```

### Modularized Construction for Deep Networks

Constructing a layer by layer deep network, such as the Google Inception, is
painful given the large number of layers. So, for such networks, we often
modularize the construction.

For example, in Google Inception network, we can first define a factory function
which chains the convolution, batch normalization and relu activation
layers together.

```python
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0),
    name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                  stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.sym.Activation(data=bn, act_type='relu', name='relu_%s%s'
                  %(name, suffix))
    return act
prev = mx.sym.Variable(name="Previos Output")
conv_comp = ConvFactory(data=prev, num_filter=64, kernel=(7,7), stride=(2, 2))
shape = {"Previos Output" : (128, 3, 28, 28)}
mx.viz.plot_network(symbol=conv_comp, shape=shape)
```

Then we can define a function that constructs an inception module based on
factory function `ConvFactory`.

```python
def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3,
                      pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1),
                      name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1),
                      name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1),
                      name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),
                      name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3),
                      pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3),
                      pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1),
                      name=('%s_proj' %  name))
    # concat
    concat = mx.sym.Concat(*[c1x1, c3x3, cd3x3, cproj],
                      name='ch_concat_%s_chconcat' % name)
    return concat
prev = mx.sym.Variable(name="Previos Output")
in3a = InceptionFactoryA(prev, 64, 64, 64, 64, 96, "avg", 32, name="in3a")
mx.viz.plot_network(symbol=in3a, shape=shape)
```

Finally, we can obtain the whole network by chaining multiple inception
modules. See a complete example
[here](https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/inception-bn.py).

### Group Multiple Symbols

To construct neural networks with multiple loss layers, we can use
`mxnet.sym.Group` to group multiple symbols together. The following example
groups two outputs:

```python
net = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
out1 = mx.sym.SoftmaxOutput(data=net, name='softmax')
out2 = mx.sym.LinearRegressionOutput(data=net, name='regression')
group = mx.sym.Group([out1, out2])
group.list_outputs()
```

## Relations to NDArray

As you can see now, both `Symbol` and `NDArray` provide multi-dimensional array
operations, such as `c=a+b` in MXNet. We briefly clarify the differences here.

The `NDArray` provides an imperative programming alike interface, in which the
computations are evaluated sentence by sentence. While `Symbol` is closer to
declarative programming, in which we first declare the computation and then
evaluate with data. Examples in this category include regular expressions and
SQL.

The pros for `NDArray`:

- Straightforward.
- Easy to work with other language features (for loop, if-else condition, ..)
  and libraries (numpy, ..).
- Easy step-by-step code debugging.

The pros for `Symbol`:

- Provides almost all functionalities of NDArray, such as +, \*, sin,
  reshape etc.
- Provides a large number of neural network related operators such as
  Convolution, Activation and BatchNorm.
- Provides automatic differentiation.
- Easy to construct and manipulate complex computations such as deep neural
  networks.
- Easy to save, load and visualization.
- Easy for the backend to optimize the computation and memory usage.

## Symbol Manipulation

One important difference of `Symbol` compared to `NDArray` is that, we first
declare the computation and then bind the computation with data to run.

In this section, we introduce the functions to manipulate a symbol directly. But
note that, most of them are wrapped by the `module` package.

### Shape and Type Inference

For each symbol, we can query it's arguments, auxiliary states and outputs.
We can also infer the output shape and type of the symbol given the known input
shape or type of some arguments, which facilitates memory allocation.

```python
arg_name = c.list_arguments()  # get the names of the inputs
out_name = c.list_outputs()    # get the names of the outputs
# infers output shape given the shape of input arguments
arg_shape, out_shape, _ = c.infer_shape(a=(2,3), b=(2,3))
# infers output type given the type of input arguments
arg_type, out_type, _ = c.infer_type(a='float32', b='float32')
{'input' : dict(zip(arg_name, arg_shape)),
 'output' : dict(zip(out_name, out_shape))}
{'input' : dict(zip(arg_name, arg_type)),
 'output' : dict(zip(out_name, out_type))}
```

### Bind with Data and Evaluate

The symbol `c` we constructed above declares what computation should be run. To
evaluate it, we first need to feed the arguments, namely free variables, with data.

We can do it by using the `bind` method, which accepts device context and
a `dict` mapping free variable names to `NDArray`s as arguments and returns an
executor. The executor provides `forward` method for evaluation and an attribute
`outputs` to get all the results.

```python
ex = c.bind(ctx=mx.cpu(), args={'a' : mx.nd.ones([2,3]),
                                'b' : mx.nd.ones([2,3])})
ex.forward()
print 'number of outputs = %d\nthe first output = \n%s' % (
           len(ex.outputs), ex.outputs[0].asnumpy())
```

We can evaluate the same symbol on GPU with different data.

```python
ex_gpu = c.bind(ctx=mx.gpu(), args={'a' : mx.nd.ones([3,4], mx.gpu())*2,
                                    'b' : mx.nd.ones([3,4], mx.gpu())*3})
ex_gpu.forward()
ex_gpu.outputs[0].asnumpy()
```

We can also use `eval` method to evaluate the symbol. It combines calls to `bind`
and `forward` methods.

```python
ex = c.eval(ctx = mx.cpu(), a = mx.nd.ones([2,3]), b = mx.nd.ones([2,3]))
print 'number of outputs = %d\nthe first output = \n%s' % (
            len(ex), ex[0].asnumpy())
```

For neural nets, a more commonly used pattern is ```simple_bind```, which
creates all of the argument arrays for you. Then you can call ```forward```,
and ```backward``` (if the gradient is needed) to get the gradient.

### Load and Save

Similar to `NDArray`, we can either serialize a `Symbol` object by using `pickle`,
or by using `save` and `load` methods directly. The only difference is that
`NDArray` uses binary format while `Symbol` uses more readable `json` format for
serialization. To convert symbol to json string, use `tojson` method.

```python
print(c.tojson())
c.save('symbol-c.json')
c2 = mx.sym.load('symbol-c.json')
c.tojson() == c2.tojson()
```

## Customized Symbol

Most operators such as `mx.sym.Convolution` and `mx.sym.Reshape` are implemented
in C++ for better performance. MXNet also allows users to write new operators
using any front-end language such as Python. It often makes the developing and
debugging much easier.

To implement an operator in Python, we just need to define the two computation
methods `forward` and `backward` with several methods for querying the
properties, such as `list_arguments` and `infer_shape`.

`NDArray` is the default type of arguments in both `forward` and `backward`.
Therefore we often also implement the computation with `NDArray` operations.
To show the flexibility of MXNet, we will demonstrate an implementation of the
`softmax` layer using NumPy. Though a NumPy based operator can be only run on CPU
and also lose some optimizations which can be applied on NDArray, it enjoys the
rich functionalities provided by NumPy.

We first create a subclass of `mx.operator.CustomOp` and then define `forward`
and `backward` methods.

```python
class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))
```

Here, we first used `asnumpy` to convert the `NDArray` inputs to `numpy.ndarray`.
Then we used `CustomOp.assign` to assign the results back to `mxnet.NDArray`.
It handles assignment based on the value of req, which can be "over write" or "add to".

Now, we can create a subclass of `mx.operator.CustomOpProp` for querying the
properties.

```python
# register this operator into MXNet by name "softmax"
@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        # softmax is a loss layer so we don't need gradient input
        # from layers above.
        super(SoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return Softmax()
```

Finally, we can use `mx.sym.Custom` with the register name to use this operator.

```python
net = mx.sym.Custom(data=prev_input, op_type='softmax')
```

## Advanced Usages

### Type Cast

By default, MXNet uses 32-bit float. But for better accuracy-performance, we can
also use a lower precision data type. For example, The Nvidia Tesla Pascal GPUs
(e.g. P100) have improved 16-bit float performance, while GTX Pascal GPUs
(e.g. GTX 1080) are fast on 8-bit integers.

To convert the data type as per the requirements, we can use `mx.sym.Cast` operator
as follows:

```python
a = mx.sym.Variable('data')
b = mx.sym.Cast(data=a, dtype='float16')
arg, out, _ = b.infer_type(data='float32')
print({'input':arg, 'output':out})

c = mx.sym.Cast(data=a, dtype='uint8')
arg, out, _ = c.infer_type(data='int32')
print({'input':arg, 'output':out})
```

### Variable Sharing

To share the contents between several symbols, we can bind these symbols with
the same array as follows:

```python
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = mx.sym.Variable('c')
d = a + b * c

data = mx.nd.ones((2,3))*2
ex = d.bind(ctx=mx.cpu(), args={'a':data, 'b':data, 'c':data})
ex.forward()
ex.outputs[0].asnumpy()
```

## How Efficient Is the Symbolic API?

In short, it is designed to be very efficient in both memory and runtime.

The major reason for introducing the Symbolic API is to bring the efficient C++
operations in powerful toolkits, such as CXXNet and Caffe, together with the
flexible dynamic NDArray operations. To maximize runtime performance and memory
utilization, all of the memory and computation resources are allocated
statically during the bind operation.

The coarse-grained operators are equivalent to CXXNet layers, which are
extremely efficient.  We also provide fine-grained operators for more flexible
composition. Because we are also performing more in-place memory allocation,
MXNet can be more memory-efficient than CXXNet, and achieves the same runtime,
with greater flexibility.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
