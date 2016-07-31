MXNet Python Overview Tutorial
==============================

This page gives a general overview of MXNet's python package. MXNet contains a
mixed flavor of elements to bake flexible and efficient
applications. There are three main concepts:

* [NDArray](#ndarray-numpy-style-tensor-computations-on-cpus-and-gpus)
  offers matrix and tensor computations on both CPU and GPU, with automatic
  parallelization
* [Symbol](#symbolic-and-automatic-differentiation) makes defining a neural
  network extremely easy, and provides automatic differentiation.
* [KVStore](#distributed-key-value-store) provides data synchronization between
  multiple GPUs and multiple machines.

You can find more information at the [Python Package Overview Page](index.md)

## NDArray: Numpy style tensor computations on CPUs and GPUs

`NDArray` is the basic operational unit in MXNet for matrix and tensor
computations. It is similar to `numpy.ndarray`, but with two additional
features:

1. **multiple device support**: all operations can be run on various devices including
CPU and GPU
2. **automatic parallelization**: all operations are automatically executed in
   parallel with each other

### Creation and Initialization

We can create an `NDArray` on either CPU or GPU:

```python
>>> import mxnet as mx
>>> a = mx.nd.empty((2, 3)) # create a 2-by-3 matrix on cpu
>>> b = mx.nd.empty((2, 3), mx.gpu()) # create a 2-by-3 matrix on gpu 0
>>> c = mx.nd.empty((2, 3), mx.gpu(2)) # create a 2-by-3 matrix on gpu 2
>>> c.shape # get shape
(2L, 3L)
>>> c.context # get device info
gpu(2)
```

They can be initialized in various ways:

```python
>>> a = mx.nd.zeros((2, 3)) # create a 2-by-3 matrix filled with 0
>>> b = mx.nd.ones((2, 3))  # create a 2-by-3 matrix filled with 1
>>> b[:] = 2 # set all elements of b to 2
```

We can copy the value from one `NDArray` to another, even if they sit on different devices:

```python
>>> a = mx.nd.ones((2, 3))
>>> b = mx.nd.zeros((2, 3), mx.gpu())
>>> a.copyto(b) # copy data from cpu to gpu
```

We can also convert `NDArray` to `numpy.ndarray`:

```python
>>> a = mx.nd.ones((2, 3))
>>> b = a.asnumpy()
>>> type(b)
<type 'numpy.ndarray'>
>>> print b
[[ 1.  1.  1.]
 [ 1.  1.  1.]]
```

and vice versa:

```python
>>> import numpy as np
>>> a = mx.nd.empty((2, 3))
>>> a[:] = np.random.uniform(-0.1, 0.1, a.shape)
>>> print a.asnumpy()
[[-0.06821112 -0.03704893  0.06688045]
 [ 0.09947646 -0.07700162  0.07681718]]
```

### Basic Operations

#### Element-wise operations

By default, `NDArray` performs element-wise operations:

```python
>>> a = mx.nd.ones((2, 3)) * 2
>>> b = mx.nd.ones((2, 3)) * 4
>>> print b.asnumpy()
[[ 4.  4.  4.]
 [ 4.  4.  4.]]
>>> c = a + b
>>> print c.asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
>>> d = a * b
>>> print d.asnumpy()
[[ 8.  8.  8.]
 [ 8.  8.  8.]]
```

If two `NDArray`s sit on different devices, we need to explicitly move them into the
same one. The following example performs computations on GPU 0:

```python
>>> a = mx.nd.ones((2, 3)) * 2
>>> b = mx.nd.ones((2, 3), mx.gpu()) * 3
>>> c = a.copyto(mx.gpu()) * b
>>> print c.asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

### Load and Save

There are two ways to save data to (load from) disks easily. The first way uses
`pickle`.  `NDArray` is pickle compatible, which means you can simply pickle the
`NDArray` as you do with `numpy.ndarray`.

```python
>>> import mxnet as mx
>>> import pickle as pkl

>>> a = mx.nd.ones((2, 3)) * 2
>>> data = pkl.dumps(a)
>>> b = pkl.loads(data)
>>> print b.asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

The second way is to directly dump a list of `NDArray` to disk in binary format.

```python
>>> a = mx.nd.ones((2,3))*2
>>> b = mx.nd.ones((2,3))*3
>>> mx.nd.save('mydata.bin', [a, b])
>>> c = mx.nd.load('mydata.bin')
>>> print c[0].asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
>>> print c[1].asnumpy()
[[ 3.  3.  3.]
 [ 3.  3.  3.]]
```

We can also dump a dict:

```python
>>> mx.nd.save('mydata.bin', {'a':a, 'b':b})
>>> c = mx.nd.load('mydata.bin')
>>> print c['a'].asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
>>> print c['b'].asnumpy()
[[ 3.  3.  3.]
 [ 3.  3.  3.]]
```

In addition, if we have set up distributed filesystems such as S3 and HDFS, we
can directly save to and load from them. For example:

```python
>>> mx.nd.save('s3://mybucket/mydata.bin', [a,b])
>>> mx.nd.save('hdfs///users/myname/mydata.bin', [a,b])
```

### Automatic Parallelization
`NDArray` can automatically execute operations in parallel. This is desirable when we
use multiple resources such as CPU, GPU cards, and CPU-to-GPU memory bandwidth.

For example, if we write `a += 1` followed by `b += 1`, and `a` is on CPU while
`b` is on GPU, then we will want to execute them in parallel to improve the
efficiency. Furthermore, data copies between CPU and GPU are also expensive, so we
hope to run them in parallel with other computations as well.

However, finding statements by eye that can be executed in parallel is hard. In the
following example, `a+=1` and `c*=3` can be executed in parallel, but `a+=1` and
`b*=3` have to be sequentially executed.

```python
a = mx.nd.ones((2,3))
b = a
c = a.copyto(mx.cpu())
a += 1
b *= 3
c *= 3
```

Luckily, MXNet can automatically resolve the dependencies and
execute operations in parallel with correctness guaranteed. In other words, we
can write a program as if it is using only a single thread, and MXNet will
automatically dispatch it to multiple devices such as multiple GPU cards or multiple
machines.

This is achieved by lazy evaluation. Any operation we write down is issued to a
internal engine, and then returned. For example, if we run `a += 1`, it
returns immediately after pushing the plus operation to the engine. This
asynchronicity allows us to push more operations to the engine, so it can determine
the read and write dependency and find the best way to execute them in
parallel.

The actual computations are finished when we want to copy the results into some
other place, such as `print a.asnumpy()` or `mx.nd.save([a])`. Therefore, if we
want to write highly parallelized code, we only need to postpone asking for
the results.


## Symbolic and Automatic Differentiation

NDArray is the basic computation unit in MXNet. Besides this, MXNet provides a
symbolic interface, named Symbol, to simplify constructing neural networks. The
symbol combines flexibility and efficiency. On the one hand, it is similar to
the network configuration in [Caffe](http://caffe.berkeleyvision.org/) and
[CXXNet](https://github.com/dmlc/cxxnet); on the other, symbols define
the computation graph in [Theano](http://deeplearning.net/software/theano/).

### Basic Composition of Symbols

The following code creates a two layer perceptron network:

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

Each symbol takes a (unique) string name. *Variable* often defines the inputs,
or free variables. Other symbols take a symbol as their input (*data*),
and may accept other hyperparameters such as the number of hidden neurons (*num_hidden*)
or the activation type (*act_type*).

The symbol can be viewed simply as a function taking several arguments whose
names are automatically generated and can be got by

```python
>>> net.list_arguments()
['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'out_label']
```

As can be seen, these arguments are the parameters needed by each symbol:

- *data* : input data needed by the variable *data*
- *fc1_weight* and *fc1_bias* : the weight and bias for the first fully connected layer *fc1*
- *fc2_weight* and *fc2_bias* : the weight and bias for the second fully connected layer *fc2*
- *out_label* : the label needed by the loss

We can also specify the automatic generated names explicitly:

```python
>>> net = mx.symbol.Variable('data')
>>> w = mx.symbol.Variable('myweight')
>>> net = mx.symbol.FullyConnected(data=net, weight=w, name='fc1', num_hidden=128)
>>> net.list_arguments()
['data', 'myweight', 'fc1_bias']
```

### More Complicated Composition

MXNet provides well-optimized symbols (see
[src/operator](https://github.com/dmlc/mxnet/tree/master/src/operator)) for
commonly used layers in deep learning. We can also easily define new operators
in python.  The following example first performs an elementwise add between two
symbols, then feeds them to the fully connected operator.

```python
>>> lhs = mx.symbol.Variable('data1')
>>> rhs = mx.symbol.Variable('data2')
>>> net = mx.symbol.FullyConnected(data=lhs + rhs, name='fc1', num_hidden=128)
>>> net.list_arguments()
['data1', 'data2', 'fc1_weight', 'fc1_bias']
```

We can also construct a symbol in a more flexible way than the single
forward composition exemplified above.

```python
>>> net = mx.symbol.Variable('data')
>>> net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
>>> net2 = mx.symbol.Variable('data2')
>>> net2 = mx.symbol.FullyConnected(data=net2, name='net2', num_hidden=128)
>>> composed_net = net(data=net2, name='compose')
>>> composed_net.list_arguments()
['data2', 'net2_weight', 'net2_bias', 'compose_fc1_weight', 'compose_fc1_bias']
```

In the above example, *net* is used as a function to apply to an existing symbol
*net*, and the resulting *composed_net* will replace the original argument *data* by
*net2* instead.

### Argument Shape Inference

Now we know how to define a symbol. Next, we can infer the shapes of
all the arguments it needs given the shape of its input data.

```python
>>> net = mx.symbol.Variable('data')
>>> net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=10)
>>> arg_shape, out_shape, aux_shape = net.infer_shape(data=(100, 100))
>>> dict(zip(net.list_arguments(), arg_shape))
{'data': (100, 100), 'fc1_weight': (10, 100), 'fc1_bias': (10,)}
>>> out_shape
[(100, 10)]
```

This shape inference can be used as an early debugging mechanism to detect
shape inconsistency.

### Bind the Symbols and Run

Now we can bind the free variables of the symbol and perform forward and backward operations.
The ```bind``` function will create a ```Executor``` that can be used to carry out the real computations.

```python
>>> # define computation graphs
>>> A = mx.symbol.Variable('A')
>>> B = mx.symbol.Variable('B')
>>> C = A * B
>>> a = mx.nd.ones(3) * 4
>>> b = mx.nd.ones(3) * 2
>>> # bind the symbol with real arguments
>>> c_exec = C.bind(ctx=mx.cpu(), args={'A' : a, 'B': b})
>>> # do forward pass calclation.
>>> c_exec.forward()
>>> c_exec.outputs[0].asnumpy()
[ 8.  8.  8.]
```
For neural nets, a more commonly used pattern is ```simple_bind```, which will create
all the argument arrays for you. Then you can call ```forward```, and ```backward``` (if gradient is needed)
to get the gradient.
```python
>>> # define computation graphs
>>> net = some symbol
>>> texec = net.simple_bind(data=input_shape)
>>> texec.forward()
>>> texec.backward()
```
The [model API](model.md) is a thin wrapper around the symbolic executors to support neural net training.

You are also strongly encouraged to read [Symbolic Configuration and Execution in Pictures](symbol_in_pictures.md),
which provides a detailed explanation of the concepts in pictures.

### How Efficient is the Symbolic API?

In short, it is designed to be very efficient in both memory and runtime.

The major reason for us to introduce the Symbolic API is to bring the efficient C++
operations in powerful toolkits such as cxxnet and caffe together with the
flexible dynamic NDArray operations. All the memory and computation resources are
allocated statically during Bind, to maximize the runtime performance and memory
utilization.

The coarse grained operators are equivalent to cxxnet layers, which are
extremely efficient.  We also provide fine grained operators for more flexible
composition. Because we are also doing more inplace memory allocation, mxnet can
be ***more memory efficient*** than cxxnet, and achieves the same runtime, with
greater flexiblity.

## Distributed Key-value Store

KVStore is a place for data sharing. We can think it as a single object shared
across different devices (GPUs and machines), where each device can push data in
and pull data out.

### Initialization

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

### Pull

We have already seen how to pull a single key-value pair. Similarly to push, we can also
pull the value into several devices by a single call.

```python
>>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
>>> kv.pull(3, out = b)
>>> print b[1].asnumpy()
[[ 6.  6.  6.]
 [ 6.  6.  6.]]
```

### Handle a list of key-value pairs

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

### Multiple machines
Base on parameter server. The `updater` will runs on the server nodes.
This section will be updated when the distributed version is ready.


<!-- ## How to Choose between APIs -->

<!-- You can mix them all as much as you like. Here are some guidelines -->
<!-- * Use Symbolic API and coarse grained operator to create established structure. -->
<!-- * Use fine-grained operator to extend parts of of more flexible symbolic graph. -->
<!-- * Do some dynamic NArray tricks, which are even more flexible, between the calls of forward and backward of executors. -->

<!-- We believe that different ways offers you different levels of flexibilty and -->
<!-- efficiency. Normally you do not need to be flexible in all parts of the -->
<!-- networks, so we allow you to use the fast optimized parts, and compose it -->
<!-- flexibly with fine-grained operator or dynamic NArray. We believe such kind of -->
<!-- mixture allows you to build the deep learning architecture both efficiently and -->
<!-- flexibly as your choice. To mix is to maximize the peformance and flexiblity. -->
