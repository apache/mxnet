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

# Manipulate data with `ndarray`

We'll start by introducing the `NDArray`, MXNet’s primary tool for storing and transforming data. If you’ve worked with `NumPy` before, you’ll notice that a NDArray is, by design, similar to NumPy’s multi-dimensional array.

## Get started

To get started, let's import the `ndarray` package (`nd` is shortform) from MXNet.

```{.python .input  n=1}
# If you hasn't install MXNet yet, you can uncomment the following line to
# install the latest stable release
# !pip install -U mxnet

from mxnet import nd
```

Next, let's see how to create a 2D array (also called a matrix) with values from two sets of numbers: 1, 2, 3 and 4, 5, 6. This might also be referred to as a tuple of a tuple of integers.

```{.python .input  n=2}
nd.array(((1,2,3),(5,6,7)))
```

We can also create a very simple matrix with the same shape (2 rows by 3 columns), but fill it with 1s.

```{.python .input  n=3}
x = nd.ones((2,3))
x
```

Often we’ll want to create arrays whose values are sampled randomly. For example, sampling values uniformly between -1 and 1. Here we create the same shape, but with random sampling.

```{.python .input  n=15}
y = nd.random.uniform(-1,1,(2,3))
y
```

You can also fill an array of a given shape with a given value, such as `2.0`.
<!-- added to improve multiplication example -->

```{.python .input  n=16}
x = nd.full((2,3), 2.0)
x
```

As with NumPy, the dimensions of each NDArray are accessible by accessing the `.shape` attribute. We can also query its `size`, which is equal to the product of the components of the shape. In addition, `.dtype` tells the data type of the stored values.

```{.python .input  n=17}
(x.shape, x.size, x.dtype)
```

## Operations

NDArray supports a large number of standard mathematical operations. Such as element-wise multiplication:

```{.python .input  n=18}
x * y
```

Exponentiation:

```{.python .input  n=23}
y.exp()
```

And grab a matrix’s transpose to compute a proper matrix-matrix product:

```{.python .input  n=24}
nd.dot(x, y.T)
```

## Indexing

MXNet NDArrays support slicing in all the ridiculous ways you might imagine accessing your data. Here’s an example of reading a particular element, which returns a 1D array with shape `(1,)`.

```{.python .input  n=25}
y[1,2]
```

Read the second and third columns from `y`.

```{.python .input  n=26}
y[:,1:3]
```

and writing to a specific element

```{.python .input  n=27}
y[:,1:3] = 2
y
```

Multi-dimensional slicing is also supported.

```{.python .input  n=28}
y[1:2,0:2] = 4
y
```

## Converting between MXNet NDArray and NumPy

Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do not share memory.

```{.python .input  n=29}
a = x.asnumpy()
(type(a), a)
```

```{.python .input  n=30}
nd.array(a)
```
