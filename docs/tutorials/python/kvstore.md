# Distributed Key-Value Store

KVStore is a place for data sharing. Think of it as a single object shared
across different devices (GPUs and computers), where each device can push data in
and pull data out.

## Initialization

Let's consider a simple example: initializing
a (`int`, `NDArray`) pair into the store, and then pulling the value out:

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

## Push, Aggregate, and Update

For any key that has been initialized, you can push a new value with the same shape to the key:

```python
    >>> kv.push(3, mx.nd.ones(shape)*8)
    >>> kv.pull(3, out = a) # pull out the value
    >>> print a.asnumpy()
    [[ 8.  8.  8.]
     [ 8.  8.  8.]]
```

The data for pushing can be stored on any device. Furthermore, you can push multiple
values into the same key, where KVStore will first sum all of these
values and then push the aggregated value:

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
`updater`. The default updater is `ASSIGN`. You can replace the default to
control how data is merged:

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

You've already seen how to pull a single key-value pair. Similarly, to push, you can
pull the value onto several devices with a single call:

```python
    >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
    >>> kv.pull(3, out = b)
    >>> print b[1].asnumpy()
    [[ 6.  6.  6.]
     [ 6.  6.  6.]]
```

## Handle a List of Key-Value Pairs

All operations introduced so far involve a single key. KVStore also provides
an interface for a list of key-value pairs. 

For a single device:

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

## Run on Multiple Machines
Based on parameter server, the `updater` runs on the server nodes.
When the distributed version is ready, we will update this section.


<!-- ## How to Choose Between APIs -->

<!-- You can mix APIs as much as you like. Here are some guidelines -->
<!-- * Use the Symbolic API and a coarse-grained operator to create  an established structure. -->
<!-- * Use a fine-grained operator to extend parts of a more flexible symbolic graph. -->
<!-- * Do some dynamic NDArray tricks, which are even more flexible, between the calls of forward and backward executors. -->

<!-- Different approaches offer you different levels of flexibility and -->
<!-- efficiency. Normally, you do not need to be flexible in all parts of the -->
<!-- network, so use the parts optimized for speed, and compose it -->
<!-- flexibly with a fine-grained operator or a dynamic NDArray. Such a -->
<!-- mixture allows you to build the deep learning architecture both efficiently and -->
<!-- flexibly as your choice.  -->

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)