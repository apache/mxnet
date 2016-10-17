# Distributed Key-value Store

KVStore is a place for data sharing. We can think it as a single object shared
across different devices (GPUs and machines), where each device can push data in
and pull data out.

## Initialization

Let's first consider a simple example: initialize
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

## Push, Aggregation, and Updater

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

For each push, KVStore combines the pushed value with the value stored using an
`updater`. The default updater is `ASSIGN`; we can replace the default to
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

## Pull

We have already seen how to pull a single key-value pair. Similarly to push, we can also
pull the value into several devices by a single call.

```python
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.pull(3, out = b)
>>> print b[1].asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

## Handle a list of key-value pairs

All operations introduced so far involve a single key. KVStore also provides
an interface for a list of key-value pairs. For a single device:

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

For multiple devices:

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

## Multiple machines
Base on parameter server. The `updater` will runs on the server nodes.
This section will be updated when the distributed version is ready.


<!-- ## How to Choose between APIs -->

<!-- You can mix them all as much as you like. Here are some guidelines -->
<!-- * Use Symbolic API and coarse grained operator to create established structure. -->
<!-- * Use fine-grained operator to extend parts of of more flexible symbolic graph. -->
<!-- * Do some dynamic NArray tricks, which are even more flexible, between the calls of forward and backward of executors. -->

<!-- We believe that different ways offers you different levels of flexibility and -->
<!-- efficiency. Normally you do not need to be flexible in all parts of the -->
<!-- networks, so we allow you to use the fast optimized parts, and compose it -->
<!-- flexibly with fine-grained operator or dynamic NArray. We believe such kind of -->
<!-- mixture allows you to build the deep learning architecture both efficiently and -->
<!-- flexibly as your choice. To mix is to maximize the performance and flexibility. -->

# Recommended Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)