# NDArray - Imperative tensor operations on CPU/GPU

In _MXNet_, `NDArray` is the core data structure for all mathematical
computations.  An `NDArray` represents a multidimensional, fixed-size homogenous
array.  If you're familiar with the scientific computing python package
[NumPy](http://www.numpy.org/), you might notice that `mxnet.ndarray` is similar
to `numpy.ndarray`.  Like the corresponding NumPy data structure, MXNet's
`NDArray` enables imperative computation.

So you might wonder, why not just use NumPy?  MXNet offers two compelling
advantages.  First, MXNet's `NDArray` supports fast execution on a wide range of
hardware configurations, including CPU, GPU, and multi-GPU machines.  _MXNet_
also scales to distributed systems in the cloud.  Second, MXNet's `NDArray`
executes code lazily, allowing it to automatically parallelize multiple
operations across the available hardware.

An `NDArray` is a multidimensional array of numbers with the same type.  We
could represent the coordinates of a point in 3D space, e.g. `[2, 1, 6]` as a 1D
array with shape (3).  Similarly, we could represent a 2D array.  Below, we
present an array with length 2 along the first axis and length 3 along the
second axis.
```
[[0, 1, 2]
 [3, 4, 5]]
```
Note that here the use of "dimension" is overloaded.  When we say a 2D array, we
mean an array with 2 axes, not an array with two components.

Each NDArray supports some important attributes that you'll often want to query:

- **ndarray.shape**: The dimensions of the array. It is a tuple of integers
  indicating the length of the array along each axis. For a matrix with `n` rows
  and `m` columns, its `shape` will be `(n, m)`.
- **ndarray.dtype**: A `numpy` _type_ object describing the type of its
  elements.
- **ndarray.size**: The total number of components in the array - equal to the
  product of the components of its `shape`
- **ndarray.context**: The device on which this array is stored, e.g. `cpu()` or
  `gpu(1)`.

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/install/index.html)
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
- GPUs - A section of this tutorial uses GPUs. If you don't have GPUs on your
machine, simply set the variable gpu_device (set in the GPUs section of this 
tutorial) to mx.cpu().

## Array Creation

There are a few different ways to create an `NDArray`.

* We can create an NDArray from a regular Python list or tuple by using the `array` function:

```python
import mxnet as mx
# create a 1-dimensional array with a python list
a = mx.nd.array([1,2,3])
# create a 2-dimensional array with a nested python list
b = mx.nd.array([[1,2,3], [2,3,4]])
{'a.shape':a.shape, 'b.shape':b.shape}
```

* We can also create an MXNet NDArray from a `numpy.ndarray` object:

```python
import numpy as np
import math
c = np.arange(15).reshape(3,5)
# create a 2-dimensional array from a numpy.ndarray object
a = mx.nd.array(c)
{'a.shape':a.shape}
```

We can specify the element type with the option `dtype`, which accepts a numpy
type. By default, `float32` is used:

```python
# float32 is used by default
a = mx.nd.array([1,2,3])
# create an int32 array
b = mx.nd.array([1,2,3], dtype=np.int32)
# create a 16-bit float array
c = mx.nd.array([1.2, 2.3], dtype=np.float16)
(a.dtype, b.dtype, c.dtype)
```

If we know the size of the desired NDArray, but not the element values, MXNet
offers several functions to create arrays with placeholder content:

```python
# create a 2-dimensional array full of zeros with shape (2,3)
a = mx.nd.zeros((2,3))
# create a same shape array full of ones
b = mx.nd.ones((2,3))
# create a same shape array with all elements set to 7
c = mx.nd.full((2,3), 7)
# create a same shape whose initial content is random and
# depends on the state of the memory
d = mx.nd.empty((2,3))
```

## Printing Arrays

When inspecting the contents of an `NDArray`, it's often convenient to first
extract its contents as a `numpy.ndarray` using the `asnumpy` function.  Numpy
uses the following layout:

- The last axis is printed from left to right,
- The second-to-last is printed from top to bottom,
- The rest are also printed from top to bottom, with each slice separated from
  the next by an empty line.

```python
b = mx.nd.arange(18).reshape((3,2,3))
b.asnumpy()
```

## Basic Operations

When applied to NDArrays, the standard arithmetic operators apply *elementwise*
calculations. The returned value is a new array whose content contains the
result.

```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((2,3))
# elementwise plus
c = a + b
# elementwise minus
d = - c
# elementwise pow and sin, and then transpose
e = mx.nd.sin(c**2).T
# elementwise max
f = mx.nd.maximum(a, c)
f.asnumpy()
```

As in `NumPy`, `*` represents element-wise multiplication. For matrix-matrix
multiplication, use `dot`.

```python
a = mx.nd.arange(4).reshape((2,2))
b = a * a
c = mx.nd.dot(a,a)
print("b: %s, \n c: %s" % (b.asnumpy(), c.asnumpy()))
```

The assignment operators such as `+=` and `*=` modify arrays in place, and thus
don't allocate new memory to create a new array.

```python
a = mx.nd.ones((2,2))
b = mx.nd.ones(a.shape)
b += a
b.asnumpy()
```

## Indexing and Slicing

The slice operator `[]` applies on axis 0.

```python
a = mx.nd.array(np.arange(6).reshape(3,2))
a[1:2] = 1
a[:].asnumpy()
```

We can also slice a particular axis with the method `slice_axis`

```python
d = mx.nd.slice_axis(a, axis=1, begin=1, end=2)
d.asnumpy()
```

## Shape Manipulation

Using `reshape`, we can manipulate any arrays shape as long as the size remains
unchanged.

```python
a = mx.nd.array(np.arange(24))
b = a.reshape((2,3,4))
b.asnumpy()
```

The `concat` method stacks multiple arrays along the first axis. Their
shapes must be the same along the other axes.

```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((2,3))*2
c = mx.nd.concat(a,b)
c.asnumpy()
```

## Reduce

Some functions, like `sum` and `mean` reduce arrays to scalars.

```python
a = mx.nd.ones((2,3))
b = mx.nd.sum(a)
b.asnumpy()
```

We can also reduce an array along a particular axis:

```python
c = mx.nd.sum_axis(a, axis=1)
c.asnumpy()
```

## Broadcast

We can also broadcast an array. Broadcasting operations, duplicate an array's
value along an axis with length 1. The following code broadcasts along axis 1:

```python
a = mx.nd.array(np.arange(6).reshape(6,1))
b = a.broadcast_to((6,4))  #
b.asnumpy()
```

It's possible to simultaneously broadcast along multiple axes. In the following example, we broadcast along axes 1 and 2:

```python
c = a.reshape((2,1,1,3))
d = c.broadcast_to((2,2,2,3))
d.asnumpy()
```

Broadcasting can be applied automatically when executing some operations,
e.g. `*` and `+` on arrays of different shapes.

```python
a = mx.nd.ones((3,2))
b = mx.nd.ones((1,2))
c = a + b
c.asnumpy()
```

## Copies

When assigning an NDArray to another Python variable, we copy a reference to the
*same* NDArray. However, we often need to make a copy of the data, so that we
can manipulate the new array without overwriting the original values.

```python
a = mx.nd.ones((2,2))
b = a
b is a # will be True
```

The `copy` method makes a deep copy of the array and its data:

```python
b = a.copy()
b is a  # will be False
```

The above code allocates a new NDArray and then assigns to *b*. When we do not
want to allocate additional memory, we can use the `copyto` method or the slice
operator `[]` instead.

```python
b = mx.nd.ones(a.shape)
c = b
c[:] = a
d = b
a.copyto(d)
(c is b, d is b)  # Both will be True
```

## Advanced Topics

MXNet's NDArray offers some advanced features that differentiate it from the
offerings you'll find in most other libraries.

### GPU Support

By default, NDArray operators are executed on CPU. But with MXNet, it's easy to
switch to another computation resource, such as GPU, when available. Each
NDArray's device information is stored in `ndarray.context`. When MXNet is
compiled with flag `USE_CUDA=1` and the machine has at least one NVIDIA GPU, we
can cause all computations to run on GPU 0 by using context `mx.gpu(0)`, or
simply `mx.gpu()`. When we have access to two or more GPUs, the 2nd GPU is
represented by `mx.gpu(1)`, etc.

**Note** In order to execute the following section on a cpu set gpu_device to mx.cpu().
```python
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.


def f():
    a = mx.nd.ones((100,100))
    b = mx.nd.ones((100,100))
    c = a + b
    print(c)
# in default mx.cpu() is used
f()
# change the default context to the first GPU
with mx.Context(gpu_device):
    f()
```

We can also explicitly specify the context when creating an array:

```python
a = mx.nd.ones((100, 100), gpu_device)
a
```

Currently, MXNet requires two arrays to sit on the same device for
computation. There are several methods for copying data between devices.

```python
a = mx.nd.ones((100,100), mx.cpu())
b = mx.nd.ones((100,100), gpu_device)
c = mx.nd.ones((100,100), gpu_device)
a.copyto(c)  # copy from CPU to GPU
d = b + c
e = b.as_in_context(c.context) + c  # same to above
{'d':d, 'e':e}
```

### Serialize From/To (Distributed) Filesystems

MXNet offers two simple ways to save (load) data to (from) disk. The first way
is to use `pickle`, as you might with any other Python objects. `NDArray` is
pickle-compatible.

```python
import pickle as pkl
a = mx.nd.ones((2, 3))
# pack and then dump into disk
data = pkl.dumps(a)
pkl.dump(data, open('tmp.pickle', 'wb'))
# load from disk and then unpack
data = pkl.load(open('tmp.pickle', 'rb'))
b = pkl.loads(data)
b.asnumpy()
```

The second way is to directly dump to disk in binary format by using the `save`
and `load` methods. We can save/load a single NDArray, or a list of NDArrays:

```python
a = mx.nd.ones((2,3))
b = mx.nd.ones((5,6))
mx.nd.save("temp.ndarray", [a,b])
c = mx.nd.load("temp.ndarray")
c
```

It's also possible to save or load a dict of NDArrays in this fashion:

```python
d = {'a':a, 'b':b}
mx.nd.save("temp.ndarray", d)
c = mx.nd.load("temp.ndarray")
c
```

The `load` and `save` methods are preferable to pickle in two respects

1. When using these methods, you can save data from within the Python interface
   and then use it later from another language's binding. For example, if we save
   the data in Python:

```python
a = mx.nd.ones((2, 3))
mx.nd.save("temp.ndarray", [a,])
```

we can later load it from R:
```
a <- mx.nd.load("temp.ndarray")
as.array(a[[1]])
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```

2. When a distributed filesystem such as Amazon S3 or Hadoop HDFS is set up, we
   can directly save to and load from it.

```
mx.nd.save('s3://mybucket/mydata.ndarray', [a,])  # if compiled with USE_S3=1
mx.nd.save('hdfs///users/myname/mydata.bin', [a,])  # if compiled with USE_HDFS=1
```

### Lazy Evaluation and Automatic Parallelization

MXNet uses lazy evaluation to achieve superior performance.  When we run `a=b+1`
in Python, the Python thread just pushes this operation into the backend engine
and then returns.  There are two benefits to this approach:

1. The main Python thread can continue to execute other computations once the
   previous one is pushed. It is useful for frontend languages with heavy
   overheads.
2. It is easier for the backend engine to explore further optimization, such as
   auto parallelization.

The backend engine can resolve data dependencies and schedule the computations
correctly. It is transparent to frontend users. We can explicitly call the
method `wait_to_read` on the result array to wait until the computation
finishes. Operations that copy data from an array to other packages, such as
`asnumpy`, will implicitly call `wait_to_read`.


```python
import time
def do(x, n):
    """push computation into the backend engine"""
    return [mx.nd.dot(x,x) for i in range(n)]
def wait(x):
    """wait until all results are available"""
    for y in x:
        y.wait_to_read()

tic = time.time()
a = mx.nd.ones((1000,1000))
b = do(a, 50)
print('time for all computations are pushed into the backend engine:\n %f sec' % (time.time() - tic))
wait(b)
print('time for all computations are finished:\n %f sec' % (time.time() - tic))
```

Besides analyzing data read and write dependencies, the backend engine is able
to schedule computations with no dependency in parallel. For example, in the
following code:

```python
a = mx.nd.ones((2,3))
b = a + 1
c = a + 2
d = b * c
```

the second and third lines can be executed in parallel. The following example
first runs on CPU and then on GPU:

```python
n = 10
a = mx.nd.ones((1000,1000))
b = mx.nd.ones((6000,6000), gpu_device)
tic = time.time()
c = do(a, n)
wait(c)
print('Time to finish the CPU workload: %f sec' % (time.time() - tic))
d = do(b, n)
wait(d)
print('Time to finish both CPU/GPU workloads: %f sec' % (time.time() - tic))
```

Now we issue all workloads at the same time. The backend engine will try to
parallel the CPU and GPU computations.

```python
tic = time.time()
c = do(a, n)
d = do(b, n)
wait(c)
wait(d)
print('Both as finished in: %f sec' % (time.time() - tic))
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
