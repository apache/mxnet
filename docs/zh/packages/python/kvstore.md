KVStore API
===========

* [基本的 Push 和 Pull 操作](#basic-push-and-pull)
* [key-value pairs 列表的接口](#interface-for-list-key-value-pairs)
* [多机]() TODO

## Basic Push and Pull

单机多卡的基本操作.

### Initialization

首先让我们来考虑一个简单的例子. 首先初始化一个 (`int`, `NDAarray`) push 到 KVstore 里, 然后再将数据   pull 下来.

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

对于任意一个被初始化的 key-value 数据, 我们可以向这个 `key` push 一个相同 shape 的数据覆盖掉原来的 value.


```python
>>> kv.push(3, mx.nd.ones(shape)*8)
>>> kv.pull(3, out = a) # pull out the value
>>> print a.asnumpy()
[[ 8.  8.  8.]
 [ 8.  8.  8.]]
```

需要做 push 操作的数据可以存储在任意的设备上. 而且, 我们可以向同一个 key 推送多份数据, KVStore 客户端会首先将这些数据做 sum 操作, 然后将聚合后的结果 push 到服务器端, 减少了数据通信.

```python
>>> gpus = [mx.gpu(i) for i in range(4)]
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.push(3, b)
>>> kv.pull(3, out = a)
>>> print a.asnumpy()
[[ 4.  4.  4.]
 [ 4.  4.  4.]]
```

对于每一个 push 操作, KVStore 将推送上来的数据通过 `updater` 定义的方式来进行更新操作. 默认的 `updater` 是 `ASSGIN`, 我们可以根据需要来替换掉这个默认的 `update`.

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

我们已经看到如何 pull 单个的 key-value 对. 类似于 push, 我们也能只用一个调用来将数据 pull 到多个设备中.

```python
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.pull(3, out = b)
>>> print b[1].asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

## Interface for list key-value pairs

我们到现在为止所介绍的所有操作都是关于一个 key. KVStore 也提供了对 key-value pair 列表的接口. 

针对单个的设备:

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

针对多个设备:

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
