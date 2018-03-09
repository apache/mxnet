
# RowSparseNDArray - NDArray for Sparse Gradient Updates

## Motivation

Many real world datasets deal with high dimensional sparse feature vectors. When learning
the weights of models with sparse datasets, the derived gradients of the weights could be sparse.

Let's say we perform a matrix multiplication of ``X``  and ``W``, where ``X`` is a 1x2 matrix, and ``W`` is a 2x3 matrix. Let ``Y`` be the matrix multiplication of the two matrices:


```python
import mxnet as mx
X = mx.nd.array([[1,0]])
W = mx.nd.array([[3,4,5], [6,7,8]])
Y = mx.nd.dot(X, W)
{'X': X, 'W': W, 'Y': Y}
```




    {'W': 
     [[ 3.  4.  5.]
      [ 6.  7.  8.]]
     <NDArray 2x3 @cpu(0)>, 'X': 
     [[ 1.  0.]]
     <NDArray 1x2 @cpu(0)>, 'Y': 
     [[ 3.  4.  5.]]
     <NDArray 1x3 @cpu(0)>}



As you can see,

```
Y[0][0] = X[0][0] * W[0][0] + X[0][1] * W[1][0] = 1 * 3 + 0 * 6 = 3
Y[0][1] = X[0][0] * W[0][1] + X[0][1] * W[1][1] = 1 * 4 + 0 * 7 = 4
Y[0][2] = X[0][0] * W[0][2] + X[0][1] * W[1][2] = 1 * 5 + 0 * 8 = 5
```

What about dY / dW, the gradient for ``W``? Let's call it ``grad_W``. To start with, the shape of ``grad_W`` is the same as that of ``W`` as we are taking the derivatives with respect to ``W``, which is 2x3. Then we calculate each entry in ``grad_W`` as follows:

```
grad_W[0][0] = X[0][0] = 1
grad_W[0][1] = X[0][0] = 1
grad_W[0][2] = X[0][0] = 1
grad_W[1][0] = X[0][1] = 0
grad_W[1][1] = X[0][1] = 0
grad_W[1][2] = X[0][1] = 0
```

As a matter of fact, you can calculate ``grad_W`` by multiplying the transpose of ``X`` with a matrix of ones:


```python
grad_W = mx.nd.dot(X, mx.nd.ones_like(Y), transpose_a=True)
grad_W
```




    
    [[ 1.  1.  1.]
     [ 0.  0.  0.]]
    <NDArray 2x3 @cpu(0)>



As you can see, row 0 of ``grad_W`` contains non-zero values while row 1 of ``grad_W`` does not. Why did that happen?
If you look at how ``grad_W`` is calculated, notice that since column 1 of ``X`` is filled with zeros, row 1 of ``grad_W`` is filled with zeros too.

In the real world, gradients for parameters that interact with sparse inputs ususally have gradients where many row slices are completely zeros. Storing and manipulating such sparse matrices with many row slices of all zeros in the default dense structure results in wasted memory and processing on the zeros. More importantly, many gradient based optimization methods such as SGD, [AdaGrad](https://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf) and [Adam](https://arxiv.org/pdf/1412.6980.pdf)
take advantage of sparse gradients and prove to be efficient and effective. 
**In MXNet, the ``RowSparseNDArray`` stores the matrix in ``row sparse`` format, which is designed for arrays of which most row slices are all zeros.**
In this tutorial, we will describe what the row sparse format is and how to use RowSparseNDArray for sparse gradient updates in MXNet.

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](https://mxnet.io/install/index.html)
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
- Basic knowledge of NDArray in MXNet. See the detailed tutorial for NDArray in [NDArray - Imperative tensor operations on CPU/GPU](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html)
- Understanding of [automatic differentiation with autograd](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html)
- GPUs - A section of this tutorial uses GPUs. If you don't have GPUs on your
machine, simply set the variable `gpu_device` (set in the GPUs section of this
tutorial) to `mx.cpu()`

## Row Sparse Format

A RowSparseNDArray represents a multidimensional NDArray using two separate 1D arrays:
`data` and `indices`.

- data: an NDArray of any dtype with shape `[D0, D1, ..., Dn]`.
- indices: a 1D int64 NDArray with shape `[D0]` with values sorted in ascending order.

The ``indices`` array stores the indices of the row slices with non-zeros,
while the values are stored in ``data`` array. The corresponding NDArray `dense` represented by RowSparseNDArray `rsp` has

``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

A RowSparseNDArray is typically used to represent non-zero row slices of a large NDArray of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

Given this two-dimension matrix:


```python
[[ 1, 2, 3],
 [ 0, 0, 0],
 [ 4, 0, 5],
 [ 0, 0, 0],
 [ 0, 0, 0]]
```

The row sparse representation would be:
- `data` array holds all the non-zero row slices of the array.
- `indices` array stores the row index for each row slice with non-zero elements.


```python
data = [[1, 2, 3], [4, 0, 5]]
indices = [0, 2]
```

`RowSparseNDArray` supports multidimensional arrays. Given this 3D tensor:


```python
[[[1, 0],
  [0, 2],
  [3, 4]],

 [[5, 0],
  [6, 0],
  [0, 0]],

 [[0, 0],
  [0, 0],
  [0, 0]]]
```

The row sparse representation would be (with `data` and `indices` defined the same as above):


```python
data = [[[1, 0], [0, 2], [3, 4]], [[5, 0], [6, 0], [0, 0]]]
indices = [0, 1]
```

``RowSparseNDArray`` is a subclass of ``NDArray``. If you query **stype** of a RowSparseNDArray,
the value will be **"row_sparse"**.

## Array Creation

You can create a `RowSparseNDArray` with data and indices by using the `row_sparse_array` function:


```python
import mxnet as mx
import numpy as np
# Create a RowSparseNDArray with python lists
shape = (6, 2)
data_list = [[1, 2], [3, 4]]
indices_list = [1, 4]
a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
# Create a RowSparseNDArray with numpy arrays
data_np = np.array([[1, 2], [3, 4]])
indices_np = np.array([1, 4])
b = mx.nd.sparse.row_sparse_array((data_np, indices_np), shape=shape)
{'a':a, 'b':b}
```




    {'a': 
     <RowSparseNDArray 6x2 @cpu(0)>, 'b': 
     <RowSparseNDArray 6x2 @cpu(0)>}



## Function Overview

Similar to `CSRNDArray`, the are several functions with `RowSparseNDArray` that behave the same way. In the code blocks below you can try out these common functions:

- **.dtype** - to set the data type
- **.asnumpy** - to cast as a numpy array for inspecting it
- **.data** - to access the data array
- **.indices** - to access the indices array
- **.tostype** - to set the storage type
- **.cast_storage** - to convert the storage type
- **.copy** - to copy the array
- **.copyto** - to copy to deep copy an existing array


## Setting Type

You can create a `RowSparseNDArray` from another specifying the element data type with the option `dtype`, which accepts a numpy type. By default, `float32` is used.


```python
# Float32 is used by default
c = mx.nd.sparse.array(a)
# Create a 16-bit float array
d = mx.nd.array(a, dtype=np.float16)
(c.dtype, d.dtype)
```




    (numpy.float32, numpy.float16)



## Inspecting Arrays

As with `CSRNDArray`, you can inspect the contents of a `RowSparseNDArray` by filling
its contents into a dense `numpy.ndarray` using the `asnumpy` function.


```python
a.asnumpy()
```




    array([[ 0.,  0.],
           [ 1.,  2.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 3.,  4.],
           [ 0.,  0.]], dtype=float32)



You can inspect the internal storage of a RowSparseNDArray by accessing attributes such as `indices` and `data`:


```python
# Access data array
data = a.data
# Access indices array
indices = a.indices
{'a.stype': a.stype, 'data':data, 'indices':indices}
```




    {'a.stype': 'row_sparse', 'data': 
     [[ 1.  2.]
      [ 3.  4.]]
     <NDArray 2x2 @cpu(0)>, 'indices': 
     [1 4]
     <NDArray 2 @cpu(0)>}



## Storage Type Conversion

You can convert an NDArray to a RowSparseNDArray and vice versa by using the `tostype` function:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type from `default` to `row_sparse`
rsp = ones.tostype('row_sparse')
# Cast the storage type from `row_sparse` to `default`
dense = rsp.tostype('default')
{'rsp':rsp, 'dense':dense}
```




    {'dense': 
     [[ 1.  1.]
      [ 1.  1.]]
     <NDArray 2x2 @cpu(0)>, 'rsp': 
     <RowSparseNDArray 2x2 @cpu(0)>}



You can also convert the storage type by using the `cast_storage` operator:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type to `row_sparse`
rsp = mx.nd.sparse.cast_storage(ones, 'row_sparse')
# Cast the storage type to `default`
dense = mx.nd.sparse.cast_storage(rsp, 'default')
{'rsp':rsp, 'dense':dense}
```




    {'dense': 
     [[ 1.  1.]
      [ 1.  1.]]
     <NDArray 2x2 @cpu(0)>, 'rsp': 
     <RowSparseNDArray 2x2 @cpu(0)>}



## Copies

You can use the `copy` method which makes a deep copy of the array and its data, and returns a new array.
We can also use the `copyto` method or the slice operator `[]` to deep copy to an existing array.


```python
a = mx.nd.ones((2,2)).tostype('row_sparse')
b = a.copy()
c = mx.nd.sparse.zeros('row_sparse', (2,2))
c[:] = a
d = mx.nd.sparse.zeros('row_sparse', (2,2))
a.copyto(d)
{'b is a': b is a, 'b.asnumpy()':b.asnumpy(), 'c.asnumpy()':c.asnumpy(), 'd.asnumpy()':d.asnumpy()}
```




    {'b is a': False, 'b.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32), 'c.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32), 'd.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32)}



If the storage types of source array and destination array do not match,
the storage type of destination array will not change when copying with `copyto` or the slice operator `[]`. The source array will be temporarily converted to desired storage type before the copy.


```python
e = mx.nd.sparse.zeros('row_sparse', (2,2))
f = mx.nd.sparse.zeros('row_sparse', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype, 'g.stype':g.stype}
```




    {'e.stype': 'row_sparse', 'f.stype': 'row_sparse', 'g.stype': 'default'}



## Retain Row Slices

You can retain a subset of row slices from a RowSparseNDArray specified by their row indices.


```python
data = [[1, 2], [3, 4], [5, 6]]
indices = [0, 2, 3]
rsp = mx.nd.sparse.row_sparse_array((data, indices), shape=(5, 2))
# Retain row 0 and row 1
rsp_retained = mx.nd.sparse.retain(rsp, mx.nd.array([0, 1]))
{'rsp.asnumpy()': rsp.asnumpy(), 'rsp_retained': rsp_retained, 'rsp_retained.asnumpy()': rsp_retained.asnumpy()}
```




    {'rsp.asnumpy()': array([[ 1.,  2.],
            [ 0.,  0.],
            [ 3.,  4.],
            [ 5.,  6.],
            [ 0.,  0.]], dtype=float32), 'rsp_retained': 
     <RowSparseNDArray 5x2 @cpu(0)>, 'rsp_retained.asnumpy()': array([[ 1.,  2.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.]], dtype=float32)}



## Sparse Operators and Storage Type Inference

Operators that have specialized implementation for sparse arrays can be accessed in ``mx.nd.sparse``. You can read the [mxnet.ndarray.sparse API documentation](http://mxnet.io/versions/master/api/python/ndarray/sparse.html) to find what sparse operators are available.


```python
shape = (3, 5)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
# A csr matrix as lhs
lhs = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
# A dense matrix as rhs
rhs = mx.nd.ones((3, 2))
# row_sparse result is inferred from sparse operator dot(csr.T, dense) based on input stypes
transpose_dot = mx.nd.sparse.dot(lhs, rhs, transpose_a=True)
{'transpose_dot': transpose_dot, 'transpose_dot.asnumpy()': transpose_dot.asnumpy()}
```




    {'transpose_dot': 
     <RowSparseNDArray 5x2 @cpu(0)>, 'transpose_dot.asnumpy()': array([[ 7.,  7.],
            [ 9.,  9.],
            [ 8.,  8.],
            [ 0.,  0.],
            [ 0.,  0.]], dtype=float32)}



For any sparse operator, the storage type of output array is inferred based on inputs. You can either read the documentation or inspect the `stype` attribute of output array to know what storage type is inferred:


```python
a = transpose_dot.copy()
b = a * 2  # b will be a RowSparseNDArray since zero multiplied by 2 is still zero
c = a + mx.nd.ones((5, 2))  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```




    {'b.stype': 'row_sparse', 'c.stype': 'default'}



For operators that don't specialize in sparse arrays, you can still use them with sparse inputs with some performance penalty.
In MXNet, dense operators require all inputs and outputs to be in the dense format.

If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily so that the dense operator can be used.

If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format.

For operators that don't specialize in sparse arrays, you can still use them with sparse inputs with some performance penalty.


```python
e = mx.nd.sparse.zeros('row_sparse', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd.stype':d.stype, 'e.stype':e.stype} # stypes of a and e will be not changed
```




    {'a.stype': 'row_sparse', 'd.stype': 'default', 'e.stype': 'row_sparse'}



Note that warning messages will be printed when such a storage fallback event happens. If you are using jupyter notebook, the warning message will be printed in your terminal console.

## Sparse Optimizers

In MXNet, sparse gradient updates are applied when weight, state and gradient are all in `row_sparse` storage.
The sparse optimizers only update the row slices of the weight and the states whose indices appear
in `gradient.indices`. For example, the default update rule for SGD optimizer is:

```
rescaled_grad = learning_rate * rescale_grad * clip(grad, clip_gradient) + weight_decay * weight
state = momentum * state + rescaled_grad
weight = weight - state
```

Meanwhile, the sparse update rule for SGD optimizer is:

```
for row in grad.indices:
    rescaled_grad[row] = learning_rate * rescale_grad * clip(grad[row], clip_gradient) + weight_decay * weight[row]
    state[row] = momentum[row] * state[row] + rescaled_grad[row]
    weight[row] = weight[row] - state[row]
```


```python
# Create weight
shape = (4, 2)
weight = mx.nd.ones(shape).tostype('row_sparse')
# Create gradient
data = [[1, 2], [4, 5]]
indices = [1, 2]
grad = mx.nd.sparse.row_sparse_array((data, indices), shape=shape)
sgd = mx.optimizer.SGD(learning_rate=0.01, momentum=0.01)
# Create momentum
momentum = sgd.create_state(0, weight)
# Before the update
{"grad.asnumpy()":grad.asnumpy(), "weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```




    {'grad.asnumpy()': array([[ 0.,  0.],
            [ 1.,  2.],
            [ 4.,  5.],
            [ 0.,  0.]], dtype=float32), 'momentum.asnumpy()': array([[ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.]], dtype=float32), 'weight.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.],
            [ 1.,  1.],
            [ 1.,  1.]], dtype=float32)}




```python
sgd.update(0, weight, grad, momentum)
# Only row 0 and row 2 are updated for both weight and momentum
{"weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```




    {'momentum.asnumpy()': array([[ 0.  ,  0.  ],
            [-0.01, -0.02],
            [-0.04, -0.05],
            [ 0.  ,  0.  ]], dtype=float32),
     'weight.asnumpy()': array([[ 1.        ,  1.        ],
            [ 0.99000001,  0.98000002],
            [ 0.95999998,  0.94999999],
            [ 1.        ,  1.        ]], dtype=float32)}



Note that both [mxnet.optimizer.SGD](https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.optimizer.SGD)
and [mxnet.optimizer.Adam](https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.optimizer.Adam) support sparse updates in MXNet.

## Advanced Topics

### GPU Support

By default, RowSparseNDArray operators are executed on CPU. In MXNet, GPU support for RowSparseNDArray is experimental
with only a few sparse operators such as cast_storage and dot.

To create a RowSparseNDArray on gpu, we need to explicitly specify the context:

**Note** If a GPU is not available, an error will be reported in the following section. In order to execute it on a cpu, set gpu_device to mx.cpu().


```python
import sys
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
try:
    a = mx.nd.sparse.zeros('row_sparse', (100, 100), ctx=gpu_device)
    a
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```



<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
