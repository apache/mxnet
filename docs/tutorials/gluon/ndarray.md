# NDArray - Scientific computing on CPU and GPU

NDArray is a tensor data structure similar to numpy's multi-dimensional array.
In addition, it supports asynchronous computation on CPU and GPU.

First, let's import MXNet:

```python
from __future__ import print_function
import numpy as np
import mxnet as mx
```

## Creating NDArray

There are many ways to create NDArray.

Construct from (nested) list:
```python
x = mx.nd.array([[1, 2, 3], [4, 5, 6]])
print(x)
```

Construct from numpy array:
```python
x_numpy = np.ones((2, 3))
x = mx.nd.array(x_numpy)
print(x)
```

Array construction routines:
```python
# create an 2x3 array of ones
x = mx.nd.ones((2, 3))
print(x)
# create an 2x3 array of zeros
x = mx.nd.zeros((2, 3))
print(x)
# create an 1d-array of 0 to 5 and reshape to 2x3
x = mx.nd.arange(6).reshape((2, 3))
print(x)
```

You can convert an NDArray to numpy array to retrieve its data with `.asnumpy()`:
```python
z = x.asnumpy()
print(z)
```

## Basic attributes

NDArray has some basic attributes that you often want to query:

**NDArray.shape**: The dimensions of the array. It is a tuple of integers
indicating the length of the array along each axis. For a matrix with `n` rows
and `m` columns, its `shape` will be `(n, m)`.

```python
print('x.shape:', x.shape)
```

**NDArray.dtype**: A `numpy` _type_ object describing the type of array
elements.

```python
print('x.dtype:', x.dtype)
```

**NDArray.size**: the total number of components in the array - equals to the
product of the components of its `shape`

```python
print('x.size:', x.size)
```

**NDArray.context**: The device on which this array is stored, e.g. `mx.cpu()`
or `mx.gpu(1)`.

```python
print('x.context:', x.context)
```

## NDArray Operations

NDArray supports a wide range of operations. Simple operations can be called
with python syntax:

```python
x = mx.nd.array([[1, 2], [3, 4]])
y = mx.nd.array([[4, 3], [2, 1]])
print(x + y)
```

You can also call operators from the `mxnet.ndarray` (or `mx.nd` for short) name space:

```python
z = mx.nd.add(x, y)
print(z)
```

You can also pass additional flags to operators:

```python
z = mx.nd.sum(x, axis=0)
print('axis=0:', z)
z = mx.nd.sum(x, axis=1)
print('axis=1:', z)
```

## Using GPU

Each NDArray lives on a `Context`. MXNet supports `mx.cpu()` for CPU and `mx.gpu(0)`,
`mx.gpu(1)`, etc for GPU. You can specify context when creating NDArray:

```python
# creates on CPU (the default).
# Replace mx.cpu() with mx.gpu(0) if you have a GPU.
x = mx.nd.zeros((2, 2), ctx=mx.cpu())
print(x)
```

```python
x = mx.nd.array([[1, 2], [3, 4]], ctx=mx.cpu())
print(x)
```

You can copy arrays between devices with `.copyto()`:

```python
# Copy x to cpu. Replace with mx.gpu(0) if you have GPU.
y = x.copyto(mx.cpu())
print(y)
```

```python
# Copy x to another NDArray, possibly on another Context.
y = mx.nd.zeros_like(x)
x.copyto(y)
print(y)
```

See the [Advanced NDArray tutorial](../basic/ndarray.md) for a more detailed
introduction to NDArray API.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
