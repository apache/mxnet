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

For some columns in ``X``, they do not have any non-zero value, therefore the gradient for the weight ``W``
will have many row slices of all zeros corresponding to the zero columns in ``X``.

```python
W = mx.nd.random_uniform(shape=(10, 2))
b = mx.nd.zeros((3, 1))
# attach a gradient placeholder for W
W.attach_grad(stype='row_sparse')
with mx.autograd.record():
    Y = mx.nd.dot(X, W) + b

Y.backward()
# the content of the gradients of `W`
{'W.grad': W.grad, 'W.grad.asnumpy()': W.grad.asnumpy()}
```

Storing and manipulating such sparse matrices with many row slices of all zeros in the default dense structure results
in wasted memory and processing on the zeros. More importantly, many gradient based optimization methods such as
SGD, [AdaGrad](https://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf) and [Adam](https://arxiv.org/pdf/1412.6980.pdf)
take advantage of sparse gradients and prove to be efficient and effective.
In MXNet, the ``RowSparseNDArray`` stores the matrix in ``row sparse`` format and provides optimizers and operators with specialized implementations.
In this tutorial, we will describe what the row sparse format is and how to use RowSparseNDArray for sparse gradient updates in MXNet.

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

A RowSparseNDArray represents a multidimensional NDArray using two separate 1D arrays:
`data` and `indices`.

- data: an NDArray of any dtype with shape `[D0, D1, ..., Dn]`.
- indices: a 1D int64 NDArray with shape `[D0]` with values sorted in ascending order.

The ``indices`` array stores the indices of the row slices with non-zeros,
while the values are stored in ``data`` array. The corresponding NDArray `dense`
represented by RowSparseNDArray `rsp` has

``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

A RowSparseNDArray is typically used to represent non-zero row slices of a large NDArray
of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

For example, the row sparse representation for this two-dimension matrix
```python
[[ 1, 2, 3],
 [ 0, 0, 0],
 [ 4, 0, 5],
 [ 0, 0, 0],
 [ 0, 0, 0]]
```
is:
```python
# `data` array holds all the non-zero row slices of the array.
data = [[1, 2, 3], [4, 0, 5]]
# `indices` array stores the row index for each row slice with non-zero elements.
indices = [0, 2]
```

RowSparseNDArray supports multidimensional arrays. The row sparse representation for this 3D tensor
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
is:
```python
# `data` array holds all the non-zero row slices of the array.
data = [[[1, 0], [0, 2], [3, 4]], [[5, 0], [6, 0], [0, 0]]]
# `indices` array stores the row index for each row slice with non-zero elements.
indices = [0, 1]
```

``RowSparseNDArray`` is a subclass of ``NDArray``. If you query **stype** of a RowSparseNDArray,
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

* We can also create a RowSparseNDArray from another specifying the element data type with the option `dtype`,
which accepts a numpy type. By default, `float32` is used.

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

# Retain Row Slices

We can retain a subset of row slices from a RowSparseNDArray specified by their row indices.

```python
data = [[1, 2], [3, 4], [5, 6]]
indices = [0, 2, 3]
rsp = mx.nd.sparse.row_sparse_array(data, indices, (5, 2))
# retain row 0 and row 1
rsp_retained = mx.nd.sparse.retain(rsp, mx.nd.array([0, 1]))
{'rsp.asnumpy()': rsp.asnumpy(), 'rsp_retained': rsp_retained, 'rsp_retained.asnumpy()': rsp_retained.asnumpy()}
```

## Sparse Operators and Storage Type Inference

* Operators that have specialized implementation for sparse arrays can be accessed in ``mx.nd.sparse``.
You can read the [mxnet.ndarray.sparse API documentation](mxnet.io/api/python/ndarray.html) to find
what sparse operators are available.

```python
shape = (3, 5)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
# a csr matrix as lhs
lhs = mx.nd.sparse.csr_matrix(data, indptr, indices, shape)
# a dense matrix as rhs
rhs = mx.nd.ones((3, 2))
# row_sparse result is inferred from sparse operator dot(csr.T, dense) based on input stypes
transpose_dot = mx.nd.sparse.dot(lhs, rhs, transpose_a=True)
{'transpose_dot': transpose_dot, 'transpose_dot.asnumpy()': transpose_dot.asnumpy()}
```

* For any sparse operator, the storage type of output array is inferred based on inputs. You can either read
the documentation or inspect the `stype` attribute of output array to know what storage type is inferred:

```python
a = transpose_dot.copy()
b = a * 2  # b will be a RowSparseNDArray since zero multiplied by 2 is still zero
c = a + 1  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```

* For operators that don't specialize in sparse arrays, we can still use them with sparse inputs with some performance penalty.
In MXNet, dense operators require all inputs and outputs to be in the dense format.
If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily so that the dense operator can be used.
If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format.
Warning messages will be printed when such a storage fallback event happens.

```python
e = mx.nd.sparse.zeros('row_sparse', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd':d, 'e':e} # stypes of a and e will be not changed
```

## Sparse Optimizers

In MXNet, sparse gradient updates are applied when weight, state and gradient are all in `row_sparse` storage.
The sparse optimizers only update the row slices of the weight and the states whose indices appear
in ``gradient.indices``. For example, the default update rule for SGD optimizer is:
```
rescaled_grad = learning_rate * rescale_grad * clip(grad, clip_gradient) + weight_decay * weight
state = momentum * state + rescaled_grad
weight = weight - state
```
while the sparse update rule for SGD optimizer is:
```
for row in grad.indices:
    rescaled_grad[row] = learning_rate * rescale_grad * clip(grad[row], clip_gradient) + weight_decay * weight[row]
    state[row] = momentum[row] * state[row] + rescaled_grad[row]
    weight[row] = weight[row] - state[row]
```

```python
# create weight
shape = (4, 2)
weight = mx.nd.ones(shape).tostype('row_sparse')
# create gradient
data = [[1, 2], [4, 5]]
indices = [1, 2]
grad = mx.nd.sparse.row_sparse_array(data, indices, shape)
sgd = mx.optimizer.SGD(learning_rate=0.01, momentum=0.01)
# create momentum
momentum = sgd.create_state(0, weight)
# before the update
{"grad.asnumpy()":grad.asnumpy(), "weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```

```python
sgd.update(0, weight, grad, momentum)
# only row 0 and row 2 are updated for both weight and momentum
{"weight.asnumpy()":weight.asnumpy(), "momentum.asnumpy()":momentum.asnumpy()}
```

Note that both [mxnet.optimizer.SGD](https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.optimizer.SGD)
and [mxnet.optimizer.Adam](https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.optimizer.Adam) support sparse update in MXNet.

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
