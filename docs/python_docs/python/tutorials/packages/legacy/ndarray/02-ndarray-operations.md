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

# NDArray Operations

## Overview
This guide will introduce you to MXNet's array operations.

This content was extracted and simplified from the gluon tutorials in
[Dive Into Deep Learning](https://d2l.ai/).

## Prerequisites
* [MXNet installed in a Python environment](/get_started).
* Python 2.7.x or Python 3.x


## Operations

NDArray supports a large number of standard mathematical operations.
Such as element-wise addition:
<!-- keeping it
easy -->

```python
import mxnet as mx
from mxnet import nd
```

```python
x = nd.ones((3, 4))
y = nd.random_normal(0, 1, shape=(3, 4))
print('x=', x)
print('y=', y)
x = x + y
print('x = x + y, x=', x)
```

Multiplication:

```python
x = nd.array([1, 2, 3])
y = nd.array([2, 2, 2])
x * y
```

And exponentiation:
<!-- with these next ones we'll just have to take your word
for it... -->

```python
nd.exp(x)
```

We can also grab a matrix's transpose to compute a proper matrix-matrix product.
<!-- because we need to do that before we have coffee every day... and you know
how those dirty, improper matrixeses can be... -->

```python
nd.dot(x, y.T)
```


## In-place operations

In the previous
example, every time we ran an operation, we allocated new memory to host its
results. For example, if we write `y = x + y`, we will dereference the matrix
that `y` used to point to and instead point it at the newly allocated memory. We
can show this using Python's `id()` function, which tells us precisely which
object a variable refers to.

<!-- dereference is something C++ people would
know but everyone else... not so much. What's the point? ;) get it? Put it in
more context as to why you care about this and why this is in front of so much
other material. Seems like an optimization topic best suited for later...
###edit### we just talked about this, so I have better context. Now I
understand, but your new reader will not. This should be covered in much more
detail, and quite possibily in its own notebook since I think it will help to
show some gotchas like you mentioned verbally. I am still leaning toward
delaying the introduction of this topic....-->

```python
print('y=', y)
print('id(y):', id(y))
y = y + x
print('after y=y+x, y=', y)
print('id(y):', id(y))
```

We can assign the result to a previously allocated array with slice notation,
e.g., `result[:] = ...`.

```python
print('x=', x)
z = nd.zeros_like(x)
print('z is zeros_like x, z=', z)
print('id(z):', id(z))
print('y=', y)
z[:] = x + y
print('z[:] = x + y, z=', z)
print('id(z) is the same as before:', id(z))
```

However, `x+y` here will still allocate a temporary buffer to store the result
before copying it to z. To make better use of memory, we can perform operations
in place, avoiding temporary buffers. To do this we specify the `out` keyword
argument every operator supports:

```python
print('x=', x, 'is in id(x):', id(x))
print('y=', y, 'is in id(y):', id(y))
print('z=', z, 'is in id(z):', id(z))
nd.elemwise_add(x, y, out=z)
print('after nd.elemwise_add(x, y, out=z), x=', x, 'is in id(x):', id(x))
print('after nd.elemwise_add(x, y, out=z), y=', y, 'is in id(y):', id(y))
print('after nd.elemwise_add(x, y, out=z), z=', z, 'is in id(z):', id(z))
```

If we're not planning to re-use ``x``, then we can assign the result to ``x``
itself. There are two ways to do this in MXNet.
1. By using slice notation x[:]
= x op y
2. By using the op-equals operators like `+=`

```python
print('x=', x, 'is in id(x):', id(x))
x += y
print('x=', x, 'is in id(x):', id(x))
```

## Slicing
MXNet NDArrays support slicing in all the ridiculous ways you might
imagine accessing your data. For a quick review:

* items start through end-1: a[start:end]
* items start through the rest of the
array: a[start:]
* items from the beginning through end-1: a[:end]
* a copy of
the whole array: a[:]

Here's an example of reading the second and third rows from `x`.

```python
x = nd.array([1, 2, 3])
print('1D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
x = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('multi-D complete array, x=', x)
s = x[1:3]
print('slicing the 2nd and 3rd elements, s=', s)
```

Now let's try writing to a specific element.

```python
print('original x, x=', x)
x[2] = 9.0
print('replaced entire row with x[2] = 9.0, x=', x)
x[0,2] = 9.0
print('replaced specific element with x[0,2] = 9.0, x=', x)
x[1:2,1:3] = 5.0
print('replaced range of elements with x[1:2,1:3] = 5.0, x=', x)
```

Multi-dimensional slicing is also supported.

```python
x = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print('original x, x=', x)
s = x[1:2,1:3]
print('plucking specific elements with x[1:2,1:3]', s)
s = x[:,:1]
print('first column with x[:,:1]', s)
s = x[:1,:]
print('first row with x[:1,:]', s)
s = x[:,3:]
print('last column with x[:,3:]', s)
s = x[2:,:]
print('last row with x[2:,:]', s)
```

## Broadcasting

You might wonder, what happens if you add a vector `y` to a
matrix `X`? These operations, where we compose a low dimensional array `y` with
a high-dimensional array `X` invoke a functionality called broadcasting. First
we'll introduce `.arange` which is useful for filling out an array with evenly
spaced data. Then we can take the low-dimensional array and duplicate it along
any axis with dimension $1$ to match the shape of the high dimensional array.
Consider the following example.

Comment (visible to demonstrate with font):
dimension one(1)? Or L(elle) or l(lil elle) or I(eye) or... ? We don't even use
the notation later, so did it need to be introduced here?

<!--Also, if you use
a shape like (3,3) you lose some of the impact and miss some errors if people
play with the values. Better to have a distinct shape so that it is more obvious
what is happening and what can break.-->

```python
x = nd.ones(shape=(3,6))
print('x = ', x)
y = nd.arange(6)
print('y = ', y)
print('x + y = ', x + y)
```

While `y` is initially of shape $6$,
MXNet infers its shape to be (1,6),
and then broadcasts along the rows to form a (3,6) matrix).
You might wonder, why did MXNet choose to interpret `y` as a (1,6) matrix and not (6,1).
That's because broadcasting prefers to duplicate along the left most axis.
We can alter this behavior by explicitly giving `y` a $2$D shape using `.reshape`.
You can also chain `.arange` and `.reshape` to do this in one step.

```python
y = y.reshape((3,1))
print('y = ', y)
print('x + y = ', x+y)
y = nd.arange(6).reshape((3,1))
print('y = ', y)
```

## Converting from MXNet NDArray to NumPy
Converting MXNet NDArrays to and from
NumPy is easy. The converted arrays do not share memory.

```python
a = x.asnumpy()
type(a)
```

```python
y = nd.array(a)
print('id(a)=', id(a), 'id(x)=', id(x), 'id(y)=', id(y))
```

## Next Up

[NDArray Contexts](03-ndarray-contexts.md)
