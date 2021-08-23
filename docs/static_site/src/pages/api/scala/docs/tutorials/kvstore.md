---
layout: page_api
title: KVStore API
permalink: /api/scala/docs/tutorials/kvstore
is_tutorial: true
tag: scala
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# KVStore API

Topics:
* [Basic Push and Pull](#basic-push-and-pull)
* [List Key-Value Pairs](#list-key-value-pairs)
* [API Reference]({{'/api/scala/docs/api/#org.apache.mxnet.KVStore'|relative_url}})


## Basic Push and Pull

Provides basic operation over multiple devices (GPUs) on a single device.

### Initialization

Let's consider a simple example. It initializes
a (`int`, `NDArray`) pair into the store, and then pulls the value out.

```scala
val kv = KVStore.create("local") // create a local kv store.
val shape = Shape(2,3)
kv.init(3, NDArray.ones(shape)*2)
val a = NDArray.zeros(shape)
kv.pull(3, out = a)
a.toArray
// Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
```

### Push, Aggregation, and Updater

For any key that's been initialized, you can push a new value with the same shape to the key, as follows:

```scala
kv.push(3, NDArray.ones(shape)*8)
kv.pull(3, out = a) // pull out the value
a.toArray
// Array[Float] = Array(8.0, 8.0, 8.0, 8.0, 8.0, 8.0)
```

The data that you want to push can be stored on any device. Furthermore, you can push multiple
values into the same key, where KVStore first sums all of these
values, and then pushes the aggregated value, as follows:

```scala
val gpus = Array(Context.gpu(0), Context.gpu(1), Context.gpu(2), Context.gpu(3))
val b = Array(NDArray.ones(shape, gpus(0)), NDArray.ones(shape, gpus(1)), \
NDArray.ones(shape, gpus(2)), NDArray.ones(shape, gpus(3)))
kv.push(3, b)
kv.pull(3, out = a)
a.toArray
// Array[Float] = Array(4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
```

For each push command, KVStore applies the pushed value to the value stored by an
`updater`. The default updater is `ASSIGN`. You can replace the default to
control how data is merged.

```scala
val updater = new MXKVStoreUpdater {
          override def update(key: Int, input: NDArray, stored: NDArray): Unit = {
            println(s"update on key $key")
            stored += input * 2
          }
          override def dispose(): Unit = {}
       }
kv.setUpdater(updater)
kv.pull(3, a)
a.toArray
// Array[Float] = Array(4.0, 4.0, 4.0, 4.0, 4.0, 4.0)
kv.push(3, NDArray.ones(shape))
// update on key 3
kv.pull(3, a)
a.toArray
// Array[Float] = Array(6.0, 6.0, 6.0, 6.0, 6.0, 6.0)
```

### Pull

You've already seen how to pull a single key-value pair. Similar to the way that you use the push command, you can
pull the value into several devices with a single call.

```scala
val b = Array(NDArray.ones(shape, gpus(0)), NDArray.ones(shape, gpus(1)),\
NDArray.ones(shape, gpus(2)), NDArray.ones(shape, gpus(3)))
kv.pull(3, outs = b)
b(1).toArray
// Array[Float] = Array(6.0, 6.0, 6.0, 6.0, 6.0, 6.0)
```

## List Key-Value Pairs

All of the operations that we've discussed so far are performed on a single key. KVStore also provides
the interface for generating a list of key-value pairs. For a single device, use the following:

```scala
val keys = Array(5, 7, 9)
kv.init(keys, Array.fill(keys.length)(NDArray.ones(shape)))
kv.push(keys, Array.fill(keys.length)(NDArray.ones(shape)))
// update on key: 5
// update on key: 7
// update on key: 9
val b = Array.fill(keys.length)(NDArray.zeros(shape))
kv.pull(keys, outs = b)
b(1).toArray
// Array[Float] = Array(3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
```
