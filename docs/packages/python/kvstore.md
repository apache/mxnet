KVStore API
===========

* [Basic Push and Pull](#basic-push-and-pull)
* [Interface for list key-value pairs](#interface-for-list-key-value-pairs)
* [Multiple machines]() TODO

## Basic Push and Pull

Basic operation over multiple devices (gpus) on a single machine.

### Initialization

Let's first consider a simple example. It initializes
a (`int`, `NDAarray`) pair into the store, and then pull the value out.

```python
>>> kv = mx.kv.create('local') # create a local kv store.
>>> shape = (2,3)
>>> kv.init(3, mx.nd.ones(shape)*2)
>>> a = mx.nd.zeros(shape)
>>> kv.pull(3, out = a)
>>> print a.asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

### Push, Aggregation, and Updater

For any key has been initialized, we can push a new value with the same shape to the key.

```python
>>> kv.push(3, mx.nd.ones(shape)*8)
>>> kv.pull(3, out = a) # pull out the value
>>> print a.asnumpy()
[[ 8.  8.  8.]
 [ 8.  8.  8.]]
```

The data for pushing can be on any device. Furthermore, we can push multiple
values into the same key, where KVStore will first sum all these
values and then push the aggregated value.

```python
>>> gpus = [mx.gpu(i) for i in range(4)]
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.push(3, b)
>>> kv.pull(3, out = a)
>>> print a.asnumpy()
[[ 4.  4.  4.]
 [ 4.  4.  4.]]
```

For each push, KVStore applies the pushed value into the value stored by a
`updater`. The default updater is `ASSGIN`, we can replace the default one to
control how data is merged.

```python
>>> def update(key, input, stored):
>>>     print "update on key: %d" % key
>>>     stored += input * 2
>>> kv._set_updater(update)
>>> kv.pull(3, out=a)
>>> print a.asnumpy()
[[ 4.  4.  4.]
 [ 4.  4.  4.]]
>>> kv.push(3, mx.nd.ones(shape))
update on key: 3
>>> kv.pull(3, out=a)
>>> print a.asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

### Pull

We already see how to pull a single key-value pair. Similar to push, we can also
pull the value into several devices by a single call.

```python
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.pull(3, out = b)
>>> print b[1].asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

## Interface for list key-value pairs

All operations introduced so far are about a single key. KVStore also provides
the interface for a list of key-value pairs. For single device:

```python
>>> keys = [5, 7, 9]
>>> kv.init(keys, [mx.nd.ones(shape)]*len(keys))
>>> kv.push(keys, [mx.nd.ones(shape)]*len(keys))
update on key: 5
update on key: 7
update on key: 9
>>> b = [mx.nd.zeros(shape)]*len(keys)
>>> kv.pull(keys, out = b)
>>> print b[1].asnumpy()
[[ 3.  3.  3.]
 [ 3.  3.  3.]]
```

For multi-devices:

```python
>>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
>>> kv.push(keys, b)
update on key: 5
update on key: 7
update on key: 9
>>> kv.pull(keys, out = b)
>>> print b[1][1].asnumpy()
[[ 11.  11.  11.]
 [ 11.  11.  11.]]
```

```eval_rst
.. raw:: html

    <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```


## API Reference

```eval_rst
.. automodule:: mxnet.kvstore
    :members:

.. raw:: html

    <script>auto_index("mxnet.kvstore");</script>
```
