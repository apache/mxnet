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

# Distributed Key-Value Store

KVStore is a place for data sharing. Think of it as a single object shared
across different devices (GPUs and computers), where each device can push data in
and pull data out.

## Initialization

Let's consider a simple example: initializing
a (`int`, `NDArray`) pair into the store, and then pulling the value out:

```python
import mxnet as mx

kv = mx.kv.create('local') # create a local kv store.
shape = (2,3)
kv.init(3, mx.nd.ones(shape)*2)
a = mx.nd.zeros(shape)
kv.pull(3, out = a)
print(a.asnumpy())
```

`[[ 2.  2.  2.],[ 2.  2.  2.]]`<!--notebook-skip-line-->

## Push, Aggregate, and Update

For any key that has been initialized, you can push a new value with the same shape to the key:

```python
kv.push(3, mx.nd.ones(shape)*8)
kv.pull(3, out = a) # pull out the value
print(a.asnumpy())
```

`[[ 8.  8.  8.],[ 8.  8.  8.]]`<!--notebook-skip-line-->

The data for pushing can be stored on any device. Furthermore, you can push multiple
values into the same key, where KVStore will first sum all of these
values and then push the aggregated value. Here we will just demonstrate pushing a list of values on CPU.
Please note summation only happens if the value list is longer than one

```python
contexts = [mx.cpu(i) for i in range(4)]
b = [mx.nd.ones(shape, ctx) for ctx in contexts]
kv.push(3, b)
kv.pull(3, out = a)
print(a.asnumpy())
```

`[[ 4.  4.  4.],[ 4.  4.  4.]]`<!--notebook-skip-line-->

For each push, KVStore combines the pushed value with the value stored using an
`updater`. The default updater is `ASSIGN`. You can replace the default to
control how data is merged:

```python
def update(key, input, stored):
    print("update on key: %d" % key)
    stored += input * 2
kv._set_updater(update)
kv.pull(3, out=a)
print(a.asnumpy())
```

`[[ 4.  4.  4.],[ 4.  4.  4.]]`<!--notebook-skip-line-->

```python
kv.push(3, mx.nd.ones(shape))
kv.pull(3, out=a)
print(a.asnumpy())
```

`update on key: 3`<!--notebook-skip-line-->

`[[ 6.  6.  6.],[ 6.  6.  6.]]`<!--notebook-skip-line-->


## Pull

You've already seen how to pull a single key-value pair. Similarly, to push, you can
pull the value onto several devices with a single call:

```python
b = [mx.nd.ones(shape, ctx) for ctx in contexts]
kv.pull(3, out = b)
print(b[1].asnumpy())
```

`[ 6.  6.  6.]],[[ 6.  6.  6.]`<!--notebook-skip-line-->

## Handle a List of Key-Value Pairs

All operations introduced so far involve a single key. KVStore also provides
an interface for a list of key-value pairs.

For a single device:

```python
keys = [5, 7, 9]
kv.init(keys, [mx.nd.ones(shape)]*len(keys))
kv.push(keys, [mx.nd.ones(shape)]*len(keys))
b = [mx.nd.zeros(shape)]*len(keys)
kv.pull(keys, out = b)
print(b[1].asnumpy())
```

`update on key: 5`<!--notebook-skip-line-->

`update on key: 7`<!--notebook-skip-line-->

`update on key: 9`<!--notebook-skip-line-->

`[[ 3.  3.  3.],[ 3.  3.  3.]]`<!--notebook-skip-line-->

For multiple devices:

```python
b = [[mx.nd.ones(shape, ctx) for ctx in contexts]] * len(keys)
kv.push(keys, b)
kv.pull(keys, out = b)
print(b[1][1].asnumpy())
```

`update on key: 5`<!--notebook-skip-line-->

`update on key: 7`<!--notebook-skip-line-->

`update on key: 9`<!--notebook-skip-line-->

`[[ 11.  11.  11.],[ 11.  11.  11.]]`<!--notebook-skip-line-->

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
* [MXNet tutorials index](/api/python/docs/tutorials/)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
