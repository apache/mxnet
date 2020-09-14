# Step 1: Manipulate data with NP on MXNet

This getting started exercise introduces the MXNet `np` package for ndarrays.
These ndarrays extend the functionality of the common NumPy ndarrays. Many NumPy
methods are also available within MXNet; therefore, we will only briefly cover
some of what is available to view more you can go [here](www.nohello.com).

## Import packages and create an array
To get started, run the following commands to import the `np` package together
with the NumPy extensions package `npx`. Together, `np` with `npx` make up the
NP on MXNet front end.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()  # Activate NumPy-like mode.
```

In this step, create a 2D array (also called a matrix). The following code
example creates a matrix with values from two sets of numbers: 1, 2, 3 and 4, 5,
6. This might also be referred to as a tuple of a tuple of integers.

```{.python .input  n=2}
np.array(((1,2,3),(5,6,7)))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "array([[1., 2., 3.],\n       [5., 6., 7.]])"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can also create a very simple matrix with the same shape (2 rows by 3
columns), but fill it with 1's.

```{.python .input  n=3}
x = np.full((2,3),1) 
x
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "array([[1., 1., 1.],\n       [1., 1., 1.]])"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Alternatively, you could use a command to generate ones.

```{.python .input  n=4}
x = np.ones((2,3)) 
x
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "array([[1., 1., 1.],\n       [1., 1., 1.]])"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can create arrays whose values are sampled randomly. For example, sampling
values uniformly between -1 and 1. The following code example creates the same
shape, but with random sampling.

```{.python .input  n=5}
y = np.random.uniform(-1,1, (2,3))
y
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "array([[0.09762704, 0.18568921, 0.43037868],\n       [0.6885315 , 0.20552671, 0.71589124]])"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

As with NumPy, the dimensions of each ndarray are shown by accessing the
`.shape` attribute. As the following code example shows, you can also query for
`size`, which is equal to the product of the components of the shape. In
addition, `.dtype` tells the data type of the stored values. As you notice when
we generate random uniform values we generate `float32` not `float64` as normal
NumPy arrays.

```{.python .input  n=6}
(x.shape, x.size, x.dtype)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "((2, 3), 6, dtype('float32'))"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You could also specifiy the datatype when you create your ndarray.

```{.python .input  n=7}
x = np.full((2,3),1,dtype='int8') 
x.dtype
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "dtype('int8')"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Versus the default of `float32`.

```{.python .input  n=8}
x = np.full((2,3),1) 
x.dtype
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "dtype('float32')"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

When we multiply, by default we use the datatype with the most precision.

```{.python .input  n=9}
x = x.astype('int8') + x.astype(int) + x.astype('float32')
x.dtype
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "dtype('float32')"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Performing operations on an array

An ndarray supports a large number of standard mathematical operations. Here are
some examples. You can perform element-wise multiplication by using the
following code example.

```{.python .input  n=10}
x * y
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "array([[0.29288113, 0.55706763, 1.291136  ],\n       [2.0655947 , 0.6165801 , 2.1476736 ]])"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can perform exponentiation by using the following code example.

```{.python .input  n=11}
np.exp(y)
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "array([[1.1025515, 1.204048 , 1.5378398],\n       [1.9907899, 1.2281718, 2.0460093]])"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can also find a matrix’s transpose to compute a proper matrix-matrix product
by using the following code example.

```{.python .input  n=12}
np.dot(x, y.T)
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "array([[2.1410847, 4.8298483],\n       [2.1410847, 4.8298483]])"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Alternatively, we could use the matrix multiplication function.

```{.python .input  n=13}
np.matmul(x,y.T)
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "array([[2.1410847, 4.8298483],\n       [2.1410847, 4.8298483]])"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can leverage built in operators, like summation.

```{.python .input  n=14}
x.sum()
```

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "array(18.)"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can also gather a mean value.

```{.python .input  n=15}
x.mean()
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "array(3.)"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can perform flatten and reshape just like you normally would in NumPy!

```{.python .input  n=16}
x.flatten()
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "array([3., 3., 3., 3., 3., 3.])"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=17}
x.reshape(6,1)
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "array([[3.],\n       [3.],\n       [3.],\n       [3.],\n       [3.],\n       [3.]])"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Indexing an array

The ndarrays support slicing in many ways you might want to access your data.
The following code example shows how to read a particular element, which returns
a 1D array with shape `(1,)`.

```{.python .input  n=18}
y[1,2]
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "array(0.71589124)"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

This example shows how to read the second and third columns from `y`.

```{.python .input  n=19}
y[:,1:3]
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "array([[0.18568921, 0.43037868],\n       [0.20552671, 0.71589124]])"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

This example shows how to write to a specific element.

```{.python .input  n=20}
y[:,1:3] = 2
y
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "array([[0.09762704, 2.        , 2.        ],\n       [0.6885315 , 2.        , 2.        ]])"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can perform multi-dimensional slicing, which is shown in the following code
example.

```{.python .input  n=21}
y[1:2,0:2] = 4
y
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "array([[0.09762704, 2.        , 2.        ],\n       [4.        , 4.        , 2.        ]])"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Converting between MXNet ndarrays and NumPy arrays

You can convert MXNet ndarrays to and from NumPy ndarrays, as shown in the
following example. The converted arrays do not share memory.

```{.python .input  n=22}
a = x.asnumpy()
(type(a), a)
```

```{.json .output n=22}
[
 {
  "data": {
   "text/plain": "(numpy.ndarray,\n array([[3., 3., 3.],\n        [3., 3., 3.]], dtype=float32))"
  },
  "execution_count": 22,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=23}
a = np.array(a)
(type(a),a)
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "(mxnet.numpy.ndarray,\n array([[3., 3., 3.],\n        [3., 3., 3.]]))"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Next Steps

Ndarrays also have some additional features which make Deep Learning possible
and efficient. Namely, differentiation, and being able to leverage GPU's.
Although we will discuss using ndarray's on gpus later, later we will discuss
autograd. Next we will abstract an additional level and talk about building
Neural Network Layers [Step 2: Create a neural network](2-nn.md)

```{.python .input}

```
