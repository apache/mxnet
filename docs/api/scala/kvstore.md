# KVStore API

Topics:
* [Basic Push and Pull](#basic-push-and-pull)
* [List Key-Value Pairs](#list-key-value-pairs)
* [API Reference](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.KVStore)


## Basic Push and Pull

Provides basic operation over multiple devices (GPUs) on a single device.

### Initialization

Let's consider a simple example. It initializes
a (`int`, `NDArray`) pair into the store, and then pulls the value out.

```scala
    scala> val kv = KVStore.create("local") // create a local kv store.
    scala> val shape = Shape(2,3)
    scala> kv.init(3, NDArray.ones(shape)*2)
    scala> val a = NDArray.zeros(shape)
    scala> kv.pull(3, out = a)
    scala> a.toArray
    Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
```

### Push, Aggregation, and Updater

For any key that's been initialized, you can push a new value with the same shape to the key, as follows:

```scala
    scala> kv.push(3, NDArray.ones(shape)*8)
    scala> kv.pull(3, out = a) // pull out the value
    scala> a.toArray
    Array[Float] = Array(8.0, 8.0, 8.0, 8.0, 8.0, 8.0)
```

The data that you want to push can be stored on any device. Furthermore, you can push multiple
values into the same key, where KVStore first sums all of these
values, and then pushes the aggregated value, as follows:

```scala
    scala> val gpus = Array(Context.gpu(0), Context.gpu(1), Context.gpu(2), Context.gpu(3))
    scala> val b = Array(NDArray.ones(shape, gpus(0)), NDArray.ones(shape, gpus(1)), \
    scala> NDArray.ones(shape, gpus(2)), NDArray.ones(shape, gpus(3)))
    scala> kv.push(3, b)
    scala> kv.pull(3, out = a)
    scala> a.toArray
    Array[Float] = Array(4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
```

For each push command, KVStore applies the pushed value to the value stored by an
`updater`. The default updater is `ASSIGN`. You can replace the default to
control how data is merged.

```scala
    scala> val updater = new MXKVStoreUpdater {
              override def update(key: Int, input: NDArray, stored: NDArray): Unit = {
                println(s"update on key $key")
                stored += input * 2
              }
              override def dispose(): Unit = {}
           }
    scala> kv.setUpdater(updater)
    scala> kv.pull(3, a)
    scala> a.toArray
    Array[Float] = Array(4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    scala> kv.push(3, NDArray.ones(shape))
    update on key 3
    scala> kv.pull(3, a)
    scala> a.toArray
    Array[Float] = Array(6.0, 6.0, 6.0, 6.0, 6.0, 6.0)
```

### Pull

You've already seen how to pull a single key-value pair. Similar to the way that you use the push command, you can
pull the value into several devices with a single call.

```scala
    scala> val b = Array(NDArray.ones(shape, gpus(0)), NDArray.ones(shape, gpus(1)),\
    scala> NDArray.ones(shape, gpus(2)), NDArray.ones(shape, gpus(3)))
    scala> kv.pull(3, outs = b)
    scala> b(1).toArray
    Array[Float] = Array(6.0, 6.0, 6.0, 6.0, 6.0, 6.0)
```

## List Key-Value Pairs

All of the operations that we've discussed so far are performed on a single key. KVStore also provides
the interface for generating a list of key-value pairs. For a single device, use the following:

```scala
    scala> val keys = Array(5, 7, 9)
    scala> kv.init(keys, Array.fill(keys.length)(NDArray.ones(shape)))
    scala> kv.push(keys, Array.fill(keys.length)(NDArray.ones(shape)))
    update on key: 5
    update on key: 7
    update on key: 9
    scala> val b = Array.fill(keys.length)(NDArray.zeros(shape))
    scala> kv.pull(keys, outs = b)
    scala> b(1).toArray
    Array[Float] = Array(3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
```

## Next Steps
* [Scala Tutorials](http://mxnet.io/tutorials/index.html#Python-Tutorials)
