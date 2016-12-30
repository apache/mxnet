# MXNet Python Symbolic API

Topics:

* [How to Compose Symbols](#overloaded-operators) introduces operator overloading of symbols.
* [Symbol Attributes](#symbol-attributes) describes how to attach attributes to symbols.
* [Serialization](#serialization) explains how to save and load symbols.
* [Executing Symbols](#executing-symbols) explains how to evaluate the symbols with data.
* [Execution API Reference](#execution-api-reference) documents the execution APIs.
* [Multiple Outputs](#multiple-outputs) explains how to configure multiple outputs.
* [Symbol Creation API Reference](#symbol-creation-api-reference) documents functions.
* [Symbol Object Document](#mxnet.symbol.Symbol) documents the Symbol object.
* [Testing Utility Reference](#testing-utility-reference) documents the testing utilities.

We also highly encouraged you to read [Symbolic Configuration and Execution in Pictures](symbol_in_pictures.md).

## How to Compose Symbols

The symbolic API provides a way to configure computation graphs.
You can configure the graphs either at the level of neural network layer operations or as fine-grained operations.

The following example configures a two-layer neural network.

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

The basic arithmetic operators (plus, minus, div, multiplication) are overloaded for
*element-wise operations* of symbols.

The following example creates a computation graph that adds two inputs together.

```python
    >>> import mxnet as mx
    >>> a = mx.symbol.Variable('a')
    >>> b = mx.symbol.Variable('b')
    >>> c = a + b
````

## Symbol Attributes

You can add attributes to symbols by providing an attribute dictionary when creating a symbol.

```python
    data = mx.sym.Variable('data', attr={'mood': 'angry'})
    op   = mx.sym.Convolution(data=data, name='conv', kernel=(1, 1),
                              num_filter=1, attr={'mood': 'so so'})
```
For proper communication with the C++ back end, both the key and values of the attribute dictionary should be strings. To retrieve the attributes, use `attr(key)` or `list_attr()`:

```
    assert data.attr('mood') == 'angry'
    assert op.list_attr() == {'mood': 'so so'}
```
For a composite symbol, you can retrieve all of the attributes associated with that symbol *and its descendants* with `list_attr(recursive=True)`. In the returned dictionary, all of the attribute names have the prefix `'symbol_name' + '_'` to prevent naming conflicts.

```python
    assert op.list_attr(recursive=True) == {'data_mood': 'angry', 'conv_mood': 'so so',
                                             'conv_weight_mood': 'so so', 'conv_bias_mood': 'so so'}
```
Notice that the `mood` attribute set for the ```Convolution``` operator is copied to `conv_weight` and `conv_bias`. They're symbols that are automatically created by the ```Convolution``` operator, and the attributes are automatically copied for them. This is especially useful for annotating context groups in model parallelism. However, if you explicitly specify the weight or bias symbols, the attributes for the host operator are *not* copied to them:

```python
    weight = mx.sym.Variable('crazy_weight', attr={'size': '5'})
    data = mx.sym.Variable('data', attr={'mood': 'angry'})
    op = mx.sym.Convolution(data=data, weight=weight, name='conv', kernel=(1, 1),
                                  num_filter=1, attr= {'mood': 'so so'})
    op.list_attr(recursive=True)
    # =>
    # {'conv_mood': 'so so',
    #  'conv_bias_mood': 'so so',
    #  'crazy_weight_size': '5',
    #  'data_mood': 'angry'}
```
As you can see, the `mood` attribute is copied to the symbol `conv_bias`, which was automatically created, but not to the manually created weight symbol `crazy_weight`.

Another way to attach attributes is to use ```AttrScope```. ```AttrScope``` automatically adds the specified attributes to all of the symbols created within that scope. For example:

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

**Naming convention**: We recommend that you choose valid variable names for attribute names. Names with double underscores (e.g., `__shape__`) are reserved for internal use. The underscore `'_'` separates a symbol name and its attributes. It's also the separator between a symbol and a variable that is automatically created by that symbol. For example, the `weight` variable that is created automatically by a ```Convolution``` operator named `conv1` is called `conv1_weight`.

**Components that use attributes**: More and more components are using symbol attributes to collect useful annotations for the computational graph. Here is a (probably incomplete) list:

- ```Variable``` uses attributes to store (optional) shape information for a variable.
- Optimizers read `__lr_mult__` and `__wd_mult__` attributes for each symbol in a computational graph. This is useful to control per-layer learning rate and decay.
- The model parallelism LSTM example uses the `__ctx_group__` attribute to divide the operators into groups that correspond to GPU devices.

## Serialization

There are two ways to save and load the symbols. You can pickle to serialize the ```Symbol``` objects.
Or, you can use the [mxnet.symbol.Symbol.save](#mxnet.symbol.Symbol.save) and [mxnet.symbol.load](#mxnet.symbol.load) functions.
The advantage of using save and load is that it's  language agnostic and cloud friendly.
The symbol is saved in JSON format. You can also directly get a JSON string using [mxnet.symbol.Symbol.tojson](#mxnet.symbol.Symbol.tojson).

The following example shows how to save a symbol to an S3 bucket, load it back, and compare two symbols using a JSON string.

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

## Executing Symbols

After you have assembled a set of symbols into a computation graph, the MXNet engine can evaluate those symbols. 
If you are training a neural network, this is typically
handled by the high-level [Model class](model.md) and the [`fit()`](model.html#mxnet.model.FeedForward.fit) function.

For neural networks used in "feed-forward", "prediction", or "inference" mode (all terms for the same
thing: running a trained network), the input arguments are the 
input data, and the weights of the neural network that were learned during training.  

To manually execute a set of symbols, you need to create an [`Executor`](#mxnet.executor.Executor) object, 
which is typically constructed by calling the [`simple_bind()`](#mxnet.symbol.Symbol.simple_bind) method on a symbol.  
For an example of this, see the sample 
[`notebook on how to use simple_bind()`](https://github.com/dmlc/mxnet/blob/master/example/notebooks/simple_bind.ipynb).



## Multiple Outputs

To group the symbols together, use the [mxnet.symbol.Group](#mxnet.symbol.Group) function.

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

After you get the ```group```, you can bind on ```group``` instead.
The resulting executor will have two outputs, one for fc1_output and one for softmax_output.

```eval_rst
    .. raw:: html

        <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```

## Symbol Creation API Reference


```eval_rst
    .. automodule:: mxnet.symbol
       :members:

    .. raw:: html

        <script>auto_index("symbol-creation-api-reference");</script>
```


## Execution API Reference


```eval_rst
    .. automodule:: mxnet.executor
       :members:

    .. raw:: html

        <script>auto_index("execution-api-reference");</script>
```


## Testing Utility Reference


```eval_rst
    .. automodule:: mxnet.test_utils
        :members:

    .. raw:: html

        <script>auto_index("testing-utility-reference");</script>
```

## Next Steps
* [Symbolic Configuration and Execution in Pictures](http://mxnet.io/api/python/symbol_in_pictures.html).
* See [IO Data Loading API](io.md) for parsing and loading data.
* See [NDArray API](ndarray.md) for vector/matrix/tensor operations.
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.

