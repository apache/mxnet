# RowSparseNDArray - NDArray for Sparse Gradient Updates

## Motivation

Many real world datasets deal with high dimensional sparse feature vectors. When learning
the weights of models with sparse datasets, the derived gradients of the weights could be sparse.
For example, let's say we learn a linear model ``Y = XW + b``, where ``X`` are sparse feature vectors:

```python
import mxnet as mx
shape = (3, 10)
# `X` only contains 4 non-zeros
data = [6, 7, 8, 9]
indptr = [0, 2, 3, 4]
indices = [0, 4, 1, 0]
X = mx.nd.sparse.csr_matrix(data, indptr, indices, shape)
# the content of `X`
X.asnumpy()
```

The gradients for the weight ``W`` will have many row slices of all zeros, since ``X`` doesn't have non-zero values for
their corresponding columns.

```python
W = mx.nd.random_uniform((10, 2))
b = mx.nd.zeros((3,))
W.attach_grad(stype='row_sparse')
with mx.autograd.record():
    Y = mx.nd.dot(X, W) + b

Y.backward()
# the content of the gradients of `W`
W.grad.asnumpy()
```

Storing and manipulating such sparse matrices in the default dense structure results
in wasted memory and processing on the zeros.
To take advantage of the such sparse matrices with many row slices of all zeros, the ``RowSparseNDArray`` in MXNet
stores the matrix in ``row sparse`` format and uses specialized algorithms in operators. In this tutorial, we will
describe what the row sparse format is and how to use RowSparseNDArray in MXNet.

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/get_started/install.html)
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
- GPUs - A section of this tutorial uses GPUs. If you don't have GPUs on your
machine, simply set the variable gpu_device (set in the GPUs section of this
tutorial) to mx.cpu().

## Row Sparse Format

A RowSparseNDArray represents a multidimensional NDArray using two separate arrays:
`data` and `indices`.

- data: an NDArray of any dtype with shape `[D0, D1, ..., Dn]`.
- indices: a 1D int64 NDArray with shape `[D0]` with values sorted in ascending order.

The `indices` stores the indices of the row slices with non-zeros,
while the values are stored in `data`. The corresponding NDArray ``dense``
represented by RowSparseNDArray ``rsp`` has

``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

A RowSparseNDArray is typically used to represent non-zero row-slices of a large NDArray
of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

For example, the row sparse representation for matrix
```
[[ 1, 2, 3],
 [ 0, 0, 0],
 [ 4, 0, 5],
 [ 0, 0, 0],
 [ 0, 0, 0]]
```
is:
```
# `data` array holds all the non-zero entries of the array in row-major order.
data = [[ 1, 2, 3], [4, 0, 5]]
# `indices` array stores the row index for each row slices with non-zero elements.
indices = [0, 2, 1]
```

TODO(haibin) k dim example

``RowSparseNDArray`` class inherits from ``NDArray`` class. If you query **stype** of a RowSparseNDArray,
the value will be **"row_sparse"**.

## Array Creation

* We can create a RowSparseNDArray with data and indices by using the `row_sparse_array` function:

```python
import mxnet as mx
import numpy as np
# create a RowSparseNDArray with python lists
shape = (6, 2)
data_list = [[1, 2], [3, 4]]
indices_list = [1, 4]
a = mx.nd.sparse.row_sparse_array(data_list, indices_list, shape)
# create a RowSparseNDArray with numpy arrays
data_np = np.array([[1, 2], [3, 4]])
indices_np = np.array([1, 4])
b = mx.nd.sparse.row_sparse_array(data_np, indices_np, shape)
{'a':a, 'b':b}
```

TODO(haibin) .array can be used to create, too

* We can specify the element data type with the option `dtype`, which accepts a numpy
type. By default, `float32` is used.

```python
# float32 is used by default
c = mx.nd.sparse.array(a)
# create a 16-bit float array
d = mx.nd.array(a, dtype=np.float16)
(c.dtype, d.dtype)
```

## Inspecting Arrays

* We can inspect the contents of a `RowSparseNDArray` by filling
its contents into a dense `numpy.ndarray` using the `asnumpy` function.

```python
a.asnumpy()
```

* We can also inspect the internal storage of a RowSparseNDArray by accessing attributes such as `indices` and `data`:

```python
# access data array
data = a.data
# access indices array
indices = a.indices
{'a.stype': a.stype, 'data':data, 'indices':indices}
```

## Storage Type Conversion

* We can convert an NDArray to a RowSparseNDArray and vice versa by using the ``tostype`` function:

```python
# create a dense NDArray
ones = mx.nd.ones((2,2))
# cast the storage type from `default` to `row_sparse`
rsp = ones.tostype('row_sparse')
# cast the storage type from `row_sparse` to `default`
dense = rsp.tostype('default')
{'rsp':rsp, 'dense':dense}
```

* We can also convert the storage type by using the ``cast_storage`` operator:

```python
# create a dense NDArray
ones = mx.nd.ones((2,2))
# cast the storage type to `row_sparse`
rsp = mx.nd.sparse.cast_storage(ones, 'row_sparse')
# cast the storage type to `default`
dense = mx.nd.sparse.cast_storage(rsp, 'default')
{'rsp':rsp, 'dense':dense}
```

## Copies

* We can use the `copy` method which makes a deep copy of the array and its data, and returns a new array.
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

* If the storage types of source array and destination array do not match,
the storage type of destination array will not change when copying with `copyto` or
the slice operator `[]`.

```python
e = mx.nd.sparse.zeros('row_sparse', (2,2))
f = mx.nd.sparse.zeros('row_sparse', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype}
```

# TODO(haibin) storage type inference see CSR tutorail
# Or, we can have a rowsparse example...


## Advanced Topics

### GPU Support

By default, RowSparseNDArray operators are executed on CPU. In MXNet, GPU support for RowSparseNDArray is experimental
with only a few sparse operators such as cast_storage and dot.

To create a RowSparseNDArray on gpu, we need to explicitly specify the context:

**Note** In order to execute the following section on a cpu set gpu_device to mx.cpu().
```python
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.

a = mx.nd.sparse.zeros('row_sparse', (100, 100), ctx=gpu_device)
a
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
