# MXNet Python Symbolic API
* [How to Compose Symbols](#how-to-compose-symbols) introduces operator overloading of symbols.
* [Symbol Attributes](#symbol-attributes) introduces how to attach attributes to symbols.
* [Serialization](#serialization) introduces how to save and load symbols.
* [Executing Symbols](#executing-symbols) introduces how to evaluate the symbols with data.
* [Execution API Reference](#execution-api-reference) gives reference to all the execution APIs.
* [Multiple Outputs](#multiple-outputs) introduces how to configure multiple outputs.
* [Symbol Creation API Reference](#module-mxnet.symbol) gives reference to all functions.
* [Symbol Object Document](#mxnet.symbol.Symbol) gives API reference to the Symbol Object.
* [Testing Utility Reference](#testing-utility-reference) gives reference to the testing utilities.

You are also highly encouraged to read [Symbolic Configuration and Execution in Pictures](symbol_in_pictures.md)
with this document.

How to Compose Symbols
----------------------
The symbolic API provides a way for you to configure the computation graphs.
You can do it in a level of neural network layer operations, as well as fine
grained operations.

The following code gives an example of two layer neural network configuration.
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

The basic arithmetic operators(plus, minus, div, multiplication) are overloaded for
***elementwise operations*** of symbols.

The following code gives an example of computation graph that add two inputs together.
```python
>>> import mxnet as mx
>>> a = mx.symbol.Variable('a')
>>> b = mx.symbol.Variable('b')
>>> c = a + b
````

Symbol Attributes
-----------------
Attributes can be attached to symbols, by providing an attribute dictionary when creating a symbol.
```python
data = mx.sym.Variable('data', attr={'mood': 'angry'})
op   = mx.sym.Convolution(data=data, name='conv', kernel=(1, 1),
                          num_filter=1, attr={'mood': 'so so'})
```
Both key and values of the attribute dictionary should be strings, in order to properly communicate with the C++ backend. The attributes can be retrieved via `attr(key)` or `list_attr()`:
```
assert data.attr('mood') == 'angry'
assert op.list_attr() == {'mood': 'so so'}
```
In the case of a composite symbol, you can also retrieve all the attributes associated with that symbol *and its descendants* via `list_attr(recursive=True)`. Note in the returned dictionary, all the attribute names are with a prefix `'symbol_name' + '_'` in order to avoid naming conflicts.
```python
assert op.list_attr(recursive=True) == {'data_mood': 'angry', 'conv_mood': 'so so',
                                        'conv_weight_mood': 'so so', 'conv_bias_mood': 'so so'}
```
Here you may noticed that the `mood` attribute we set for the ```Convolution``` operator is copied to `conv_weight` and `conv_bias`. Those are symbols automatically created by the ```Convolution``` operator, and the attributes are also automatically copied for them. This is intentional and is especially useful for annotation of context groups in model parallelism. However, if the weight or bias symbol are explicitly created by the user, then the attributes for the host operator will *not* be copied to them:
```python
weight = mx.sym.Variable('crazy_weight', attr={'size': '5'})
data = mx.sym.Variable('data', attr={'mood': 'angry'})
op = mx.sym.Convolution(data=data, weight=weight, name='conv', kernel=(1, 1),
                              num_filter=1, attr={'mood': 'so so'})
op.list_attr(recursive=True)
# =>
# {'conv_mood': 'so so',
#  'conv_bias_mood': 'so so',
#  'crazy_weight_size': '5',
#  'data_mood': 'angry'}
```
As you can see, the `mood` attribute is copied to the automatically created symbol `conv_bias`, but not to the manually created weight symbol `crazy_weight`.

Another way of attaching attributes is to use ```AttrScope```. An ```AttrScope``` will automatically add the specified attributes to all the symbols created within that scope. For example:
```python
data = mx.symbol.Variable('data')
with mx.AttrScope(group='4', data='great'):
    fc1 = mx.symbol.Activation(data, act_type='relu')
    with mx.AttrScope(init_bias='0.0'):
        fc2 = mx.symbol.FullyConnected(fc1, num_hidden=10, name='fc2')
assert fc1.attr('data') == 'great'
assert fc2.attr('data') == 'great'
assert fc2.attr('init_bias') == '0.0'
```

**Naming convention**: it is recommended to choose the attribute names to be valid variable names. Names with double under-scope (e.g. `__shape__`) are reserved for internal use. The slash `'_'` is the character used to separate a symbol name and its attributes, as well as the separator between a symbol and a variable that is automatically created by that symbol. For example, the `weight` variable created automatically by a ```Convolution``` operator named `conv1` will be called `conv1_weight`.

**Components that uses attributes**: more and more components are using symbol attributes to collect useful annotations for the computational graph. Here is a (probably incomplete) list:

- ```Variable``` use attributes to store (optional) shape information for a variable.
- Optimizers will read `lr_mult` and `wd_mult` attributes for each symbol in a computational graph. This is useful to control per-layer learning rate and decay.
- The model parallelism LSTM example uses `ctx_group` attribute to divide the operators into different groups corresponding to different GPU devices.

Serialization
-------------
There are two ways to save and load the symbols. You can use pickle to serialize the ```Symbol``` objects.
Alternatively, you can use [mxnet.symbol.Symbol.save](#mxnet.symbol.Symbol.save) and [mxnet.symbol.load](#mxnet.symbol.load), functions.
The advantage of using save and load is that it is language agnostic, and also being cloud friendly.
The symbol is saved in JSON format. You can also directly get a JSON string using [mxnet.symbol.Symbol.tojson](#mxnet.symbol.Symbol.tojson)

The following code gives an example of saving a symbol to S3 bucket, load it back and compare two symbols using JSON string.
```python
>>> import mxnet as mx
>>> a = mx.symbol.Variable('a')
>>> b = mx.symbol.Variable('b')
>>> c = a + b
>>> c.save('s3://my-bucket/symbol-c.json')
>>> c2 = mx.symbol.load('s3://my-bucket/symbol-c.json')
>>> c.tojson() == c2.tojson()
True
```

Executing Symbols
-----------------
Once you have assembled a set of symbols into a computation graph, the MXNet engine can evaluate those symbols. 
If you are training a neural network, this is typically
all handled by the high level [Model class](model.md) and the [`fit()`](model.html#mxnet.model.FeedForward.fit) function.

For neural networks used in "feed-forward", "prediction", or "inference" mode (all different terms for the same
thing: running a trained network), the input arguments will be the 
input data, and the weights of the neural network that were learned during training.  

To manually execute a set of symbols, you need to create an [`Executor`](#mxnet.executor.Executor) object, 
which is typically constructed by calling the [`simple_bind()`](#mxnet.symbol.Symbol.simple_bind) method on a symbol.  
For an example of this, see the sample 
[`notebook on how to use simple_bind()`](https://github.com/dmlc/mxnet/blob/master/example/notebooks/simple_bind.ipynb)
To see how to manually 


Multiple Outputs
---------------
You can use [mxnet.symbol.Group](#mxnet.symbol.Group) function to group the symbols together.

```python
>>> import mxnet as mx
>>> net = mx.symbol.Variable('data')
>>> fc1 = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
>>> net = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
>>> net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=64)
>>> out = mx.symbol.SoftmaxOutput(data=net, name='softmax')
>>> group = mx.symbol.Group([fc1, out])
>>> group.list_outputs()
['fc1_output', 'softmax_output']
```

After you get the ```group```, you can go ahead and bind on ```group``` instead,
and the resulting executor will have two outputs, one for fc1_output and one for softmax_output.

```eval_rst
.. raw:: html

    <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```

Symbol Creation API Reference
-----------------------------

```eval_rst
.. automodule:: mxnet.symbol
    :members:

.. raw:: html

    <script>auto_index("mxnet.symbol");</script>
```


Execution API Reference
-----------------------

```eval_rst
.. automodule:: mxnet.executor
    :members:

.. raw:: html

    <script>auto_index("mxnet.executor");</script>
```


Testing Utility Reference
-------------------------

```eval_rst
.. automodule:: mxnet.test_utils
    :members:

.. raw:: html

    <script>auto_index("mxnet.test_utils");</script>
```

# Recommended Next Steps
* [Symbolic Configuration and Execution in Pictures](http://mxnet.io/api/python/symbol_in_pictures.html)
* [IO Data Loading API](io.md) for parsing and loading data
* [NDArray API](ndarray.md) for vector/matrix/tensor operations
* [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training

