MXNet Python Guide
==================
This page gives a general overvie of MXNet python package.
MXNet contains a mixed flavor of elements you might need to bake flexible and efficient applications.
There are two major components in MXNet:
* Numpy style [NArray API](#getting-started-with-narray) that
offers matrix and tensor computations on both CPU and GPU, and atomatically parallelize the computation for you;
* [Symbolic API](#symbolic-api-and-differentiation) that allows you define a computation graph(configure a neural network),
  and automatically gradient for you.

We aim to cover a taste of each flavor in this page.
You are welcomed to also take look at the API reference page Listed in below, or direct skip to next section.

List of Python Documents
------------------------
* [NArray API](narray.md)
* [Symbolic API](symbol.md)

Getting Started with NArray
---------------------------
The basic operation unit in MXNet is ```NArray```.
NArray is basically same as ```numpy.ndarray``` in python,
with two additional features: ***multiple device computation*** and ***automatic parallelism***.

### Create NArray and Basics
You can create ```NArray``` in both GPU and GPU, and get the shape of NArray.
```python
import mxnet as mx

cpu_array = mx.narray.create((10, 10))
gpu_array = mx.narray.create((10, 10), mx.Context('gpu', 0))
print(cpu_array.shape)
```
If the NArray sits on CPU, we can get a ```numpy.ndarray``` equivalence as follows
```python
numpy_array = cpu_array.numpy
cpu_array.numpy[:] = 10
print(cpu_array.numpy)
```
Of course, NArray itself support basic computations such as elementwise operations.
The following example adds two narray together, and creates a new ```NArray```.

```python
a = mx.narray.create((10, 10))
b = mx.narray.create((10, 10))
a.numpy[:] = 10
b.numpy[:] = 20
c = a + b
print(c.numpy)
```

Now we know how to create and manipulate NArrays. If we have some data on CPU,
how can we make use of GPU and help us to speedup computations? You can use
the copy function to copy NArray between devices, like the following example.
```python
cpu_array = mx.narray.create((10, 10))
gpu_array = mx.narray.create((10, 10), mx.Context('gpu', 0))
cpu_array.numpy[:] = 1

# copy to an allocated GPU array
cpu_array.copyto(gpu_array)

# create a new copy of NArray on GPU 0
gpu_array2 = cpu_array.copyto(mx.Context('gpu', 0))

# do some operations on GPU
gpu_array = gpu_array + gpu_array2

# copy back to CPU
gpu_array.copyto(cpu_array)

# print the result
print(cpu_array.numpy)
```

In common workflow, it is encouraged to copy the data into a GPU NArray,
do as much as computation as you can, and copy it back to CPU.

### Automatically Parallelizing Computation
So far you have learnt the basics of NArray, hope you like the flavor so far.
In machine learning scenarios, it is very common that we can have parallel
computation path, where computation can run concurrently. For example, in the following code,
```a = a + 1``` and ```b = b + 1``` can run in parallel.
```python
a = mx.narray.create((10, 10))
b = mx.narray.create((10, 10))
a.numpy[:] = 10
b.numpy[:] = 20
a = a + 1
b = b + 1
c = a + b
```
This might be a toy example, but real usecases exists, for example when we want to parallel run
neural net computation on four GPUs. Sometimes we can do this by manually creating threads
and have each of the thread drive the computation.

However, it is really non-trivial task to synchronize between threads.
Even in the toy example like the above case, we need to wait both operations on a and b to complete
until we can execute ```c = a + b```.

There are even more subtle cases, for example, in the following case, ```b.copyto(a)``` need to wait
the ```c  = a + 1``` to finish. Otherwise we might get different result for c.
```python
a = mx.narray.create((10, 10))
b = mx.narray.create((10, 10))
c = a + 1
b.copyto(a)
```

As you can see, it is really hard to write parallel programs, and really hard to reason what can be parallelized.
So normally people just give up and stay with single threaded programs.
Luckily, mxnet does the parallelism ***automatically*** and ***correctly*** for you.

So when you write the program, you can write them in normal way,
and mxnet will try to run the computation as soon as the dependency get resolved in a parallel way.
One thing that you need to know about though, is that that mxnet's computation is ***asynchronizely issued***.
So the script will immediately return, but the result may not yet be ready.

To wait the computation to finish, you can call ```wait``` function on the NArray.
The ```wait``` function is called in ```NArray.numpy```, so the result is always synchronized
and you do not need to worry about doing anything wrong.
Due to the same ready, it is adviced to use NArray as much as possible to gain parallelism.

```python
a = mx.narray.create((10, 10))
a.numpy[:] = 10
a = a + 1
a.wait()
```
So far the examples are on CPU. Of course same thing works for GPU and multiple GPUs,
for example the following snippet copies the data into two GPUs, runs the computation
and copy things back.

```python
a = mx.narray.create((10, 10))
a.numpy[:] = 10
a_gpu1 = a.copyto(mx.Context('gpu', 0))
a_gpu1 = a_gpu1 + 1
a_gpu2 = a.copyto(mx.Context('gpu', 1))
a_gpu2 = a_gpu2 + 1

print(a_gpu1.copyto(mx.Context('cpu')).numpy)
print(a_gpu2.copyto(mx.Context('cpu')).numpy)
```
As usual, mxnet will automatically do all the parallelization for you, to give you maximum efficiency.

Symbolic API and Differentiation
--------------------------------
Now you have seen the power of NArray of MXNet. It seems to be interesting and we are ready to build some real deep learning.
Hmm, this seems to be really exciting, but wait, do we need to build things from scratch?
It seems that we need to re-implement all the layers in deep learning toolkits such as [CXXNet](https://github.com/dmlc/cxxnet) in NArray?
Well, you do not have to. There is a Symbolic API in MXNet that readily helps you to do all these.

More importantly, the Symbolic API is designed to bring in the advantage of C++ static layers(operators) to ***maximumly optimizes the performance and memory*** that is even better than CXXNet. Sounds exciting? Let us get started on this.

### Creating Symbols
A common way to create a neural network is to create it via some way of configuration file or API.
The following code creates a configuration two layer perceptrons.
```python
import mxnet.symbol as sym

data = sym.Variable('data')
net = sym.FullyConnected(data=data, name='fc1', num_hidden=128)
net = sym.Activation(data=net, name='relu1', act_type="relu")
net = sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = sym.Softmax(data=net, name = 'sm')
```
If you are familiar with tools such as cxxnet or caffe, the ```Symbol``` object is like configuration files
that configures the network structure. If you are more familiar with tools like theano, the ```Symbol```
object something that defines the computation graph. Basically, it creates a computation graph
that defines the forward pass of neural network.

The Configuration API allows you to define the computation graph via compositions.
If you have not used symbolic configuration tools like theano before, one thing to
note is that the ```net``` can also be viewed as function that have input arguments.

You can get the list of arguments by calling ```Symbol.list_arguments```.
```python
>>> net.list_arguments()
['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias']
```
In our example, you can find that the arguments contains the parameters in each layer, as well as input data.
One thing that worth noticing is that the argument names like ```fc1_weight``` are automatically generated because
it was not specified in creation of fc1.
You can also specify it explicitly, like the following code.
```python
>>> import mxnet.symbol as sym
>>> data = sym.Variable('data')
>>> w = sym.Variable('myweight')
>>> net = sym.FullyConnected(data=data, weight=w,
                             name='fc1', num_hidden=128)
>>> net.list_arguments()
['data', 'myweight', 'fc1_bias']
```

### More Complicated Composition
In the previous example, Symbols are constructed in a forward compositional way.

TODO

### Shape Inference
Now we have defined the computation graph. In the next section, we are going to bind them to execution devices and
really run these computations. But before doing so, we need to figure out the shapes of the arguments, specifically,
the shape of all the weights, bias and outputs.

You can use ```Symbol.infer_shape``` to do that. Basically, shape inference allows you to shapes of arguments that you know,
and it will try to infer the shapes of all arguments and outputs.
```python
>>> import mxnet.symbol as sym
>>> data = sym.Variable('data')
>>> net = sym.FullyConnected(data=data, name='fc1',
                             num_hidden=10)
>>> arg_shape, out_shape = net.infer_shape(data=(100, 100))
>>> dict(zip(net.list_arguments(), arg_shape))
{'data': (100, 100), 'fc1_weight': (10, 100), 'fc1_bias': (10,)}
>>> out_shape
[(100, 10)]
```
In common practice, you only need to provide the shape of input data, and it
will automatically infers the shape of all the parameters.
You can always also provide more shape information, such as shape of weights.
The ```infer_shape``` will detect if there is inconsitency in the shapes,
and raise an Error if some of them are inconsistent.

### Bind the Symbols
Symbols are configuration objects that represents a computation graph(a configuration of neuralnet)
that do not


### How Efficient is Symbolic API
In short, they design to be very efficienct in both memory and runtime.

The major reason for us to introduce Symbolic API, is to bring the efficient
C++ operations in powerful toolkits such as cxxnet and caffe together with the flexible
dynamic NArray operations. All the memory and computation resources are allocated statically during Bind,
to maximize the runtime performance and memory utilization.

The coarse grained operators are equivalent to cxxnet layers, which are extremely efficient.
We also provide fine grained operators for more flexible composition. Because we are also doing more inplace
memory allocation, mxnet can be ***more memory efficient*** than cxxnet, and gets to same runtime, with greater flexiblity.

How to Choose between APIs
--------------------------
You can mix them all as much as you like. Here are some guidelines
* Use Symbolic API and coarse grained operator to create established structure.
* Use fine-grained operator to extend parts of of more flexible symbolic graph.
* Do some dynamic NArray tricks, which are even more flexible, between the calls of forward and backward of executors.

We believe that different ways offers you different levels of flexibilty and efficiency. Normally you do not need to
be flexible in all parts of the networks, so we allow you to use the fast optimized parts,
and compose it flexibly with fine-grained operator or dynamic NArray. We believe such kind of mixture allows you to build
the deep learning architecture both efficiently and flexibly as your choice. To mix is to maximize the peformance and flexiblity.