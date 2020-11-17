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

# Step 1: Manipulate data with NP on MXNet

This getting started exercise introduces the MXNet `np` package for ndarrays.
These ndarrays extend the functionality of the common NumPy ndarrays, by adding
support for gpu's and by adding auto-differentiation with autograd. Now, many
NumPy methods are available within MXNet; therefore, we will only briefly cover
some of what is available.

## Import packages and create an array
To get started, run the following commands to import the `np` package together
with the NumPy extensions package `npx`. Together, `np` with `npx` make up the
NP on MXNet front end.

```python
import mxnet as mx
from mxnet import np, npx
npx.set_np()  # Activate NumPy-like mode.
```

In this step, create a 2D array (also called a matrix). The following code
example creates a matrix with values from two sets of numbers: 1, 2, 3 and 4, 5,
6. This might also be referred to as a tuple of a tuple of integers.

```python
np.array(((1, 2, 3), (5, 6, 7)))
```

You can also create a very simple matrix with the same shape (2 rows by 3
columns), but fill it with 1's.

```python
x = np.full((2, 3), 1) 
x
```

Alternatively, you could use the following array creation routine.

```python
x = np.ones((2, 3)) 
x
```

You can create arrays whose values are sampled randomly. For example, sampling
values uniformly between -1 and 1. The following code example creates the same
shape, but with random sampling.

```python
y = np.random.uniform(-1, 1, (2, 3))
y
```

As with NumPy, the dimensions of each ndarray are shown by accessing the
`.shape` attribute. As the following code example shows, you can also query for
`size`, which is equal to the product of the components of the shape. In
addition, `.dtype` tells the data type of the stored values. As you notice when
we generate random uniform values we generate `float32` not `float64` as normal
NumPy arrays.

```python
(x.shape, x.size, x.dtype)
```

You could also specifiy the datatype when you create your ndarray.

```python
x = np.full((2, 3), 1, dtype="int8") 
x.dtype
```

Versus the default of `float32`.

```python
x = np.full((2, 3), 1) 
x.dtype
```

When we multiply, by default we use the datatype with the most precision.

```python
x = x.astype("int8") + x.astype(int) + x.astype("float32")
x.dtype
```

## Performing operations on an array

A ndarray supports a large number of standard mathematical operations. Here are
some examples. You can perform element-wise multiplication by using the
following code example.

```python
x * y
```

You can perform exponentiation by using the following code example.

```python
np.exp(y)
```

You can also find a matrix’s transpose to compute a proper matrix-matrix product
by using the following code example.

```python
np.dot(x, y.T)
```

Alternatively, you could use the matrix multiplication function.

```python
np.matmul(x, y.T)
```

You can leverage built in operators, like summation.

```python
x.sum()
```

You can also gather a mean value.

```python
x.mean()
```

You can perform flatten and reshape just like you normally would in NumPy!

```python
x.flatten()
```

```python
x.reshape(6, 1)
```

## Indexing an array

The ndarrays support slicing in many ways you might want to access your data.
The following code example shows how to read a particular element, which returns
a 1D array with shape `(1,)`.

```python
y[1, 2]
```

This example shows how to read the second and third columns from `y`.

```python
y[:, 1:3]
```

This example shows how to write to a specific element.

```python
y[:, 1:3] = 2
y
```

You can perform multi-dimensional slicing, which is shown in the following code
example.

```python
y[1:2, 0:2] = 4
y
```

## Converting between MXNet ndarrays and NumPy arrays

You can convert MXNet ndarrays to and from NumPy ndarrays, as shown in the
following example. The converted arrays do not share memory.

```python
a = x.asnumpy()
(type(a), a)
```

```python
a = np.array(a)
(type(a), a)
```

Additionally, you can move them to different GPU contexts. You will dive more
into this later, but here is an example for now.

```python
a.copyto(mx.gpu(0))
```

## Next Steps

Ndarrays also have some additional features which make Deep Learning possible
and efficient. Namely, differentiation, and being able to leverage GPU's.
Another important feature of ndarrays that we will discuss later is 
autograd. But first, we will abstract an additional level and talk about building
Neural Network Layers [Step 2: Create a neural network](2-create-nn.md)
