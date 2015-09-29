MXNet Python Symbolic API
=========================
* [How to Commpose Symbols](#overloaded-operators) introduces operator overloading of symbols
* [Serialization](#serialization) introduces how to save and load symbols.
* [Multiple Outputs](#multiple-outputs) introduces how to configure multiple outputs
* [Symbol Creation API Reference](#symbol-creationapi-reference) gives reference to all functions.
* [Symbol Object Document](#mxnet.symbol.Symbol) gives API reference to the Symbol Object
* [Execution API Reference](#execution-api-reference) tell us on what executor can do.

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
>>> net = mx.symbol.Softmax(data=net, name='out')
>>> type(net)
<class 'mxnet.symbol.Symbol'>
```

The basic arithematic operators(plus, minus, div, multiplication) are overloaded for
***elementwise operations*** of symbols.

The following code gives an example of computation graph that add two inputs together.
```python
>>> import mxnet as mx
>>> a = mx.symbol.Variable('a')
>>> b = mx.symbol.Variable('b')
>>> c = a + b
````

Serialization
-------------
There are two ways to save and load the symbols. You can use pickle to serialize the ```Symbol``` objects.
Alternatively, you can use [mxnet.symbol.Symbol.save](#mxnet.symbol.Symbol.save) and [mxnet.symbol.load](#mxnet.symbol.load), functions.
The advantage of using save and load is that it is language agnostic, and also being cloud friendly.
The symbol is saved in json format. You can also directly get a json string using [mxnet.symbol.Symbol.tojson](#mxnet.symbol.Symbol.tojson)

The following code gives an example of saving a symbol to S3 bucket, load it back and compare two symbols using json string.
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

Multiple Ouputs
---------------
You can use [mxnet.symbol.Group](#mxnet.symbol.Group) function to group the symbols together.

```python
>>> import mxnet as mx
>>> net = mx.symbol.Variable('data')
>>> fc1 = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
>>> net = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
>>> net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=64)
>>> out = mx.symbol.Softmax(data=net, name='softmax')
>>> group = mx.symbol.Group([fc1, out])
>>> group.list_outputs()
['fc1_output', 'softmax_output']
```

After you get the ```group```, you can go ahead and bind on ```group``` instead,
and the resulting executor will have two outputs, one for fc1_output and one for softmax_output.

Symbol Creation API Reference
-----------------------------
This section contains the reference to all API functions.
Before you get started, you can check the list of functions in the following table.

```eval_rst
.. autosummary::
   :nosignatures:

   mxnet.symbol.load
   mxnet.symbol.load_json
   mxnet.symbol.Group
   mxnet.symbol.Variable
   mxnet.symbol.Activation
   mxnet.symbol.BatchNorm
   mxnet.symbol.Concat
   mxnet.symbol.Convolution
   mxnet.symbol.Dropout
   mxnet.symbol.ElementWiseSum
   mxnet.symbol.Flatten
   mxnet.symbol.FullyConnected
   mxnet.symbol.LRN
   mxnet.symbol.LeakyReLU
   mxnet.symbol.Pooling
   mxnet.symbol.Reshape
   mxnet.symbol.Softmax
```

```eval_rst
.. automodule:: mxnet.symbol
    :members:
```


Execution API Reference
-----------------------

```eval_rst
.. automodule:: mxnet.executor
    :members:
```
