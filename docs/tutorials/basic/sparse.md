# CSRNDArray - NDArray in Compressed Sparse Row Storage Format

Many real world datasets deal with high dimensional sparse feature vectors. For instance,
in a recommendation system, the number of categories and users is in the order of millions,
while most users only made a few purchases, leading to feature vectors with high sparsity
(i.e. most of the elements are zeros).

Storing and manipulating such large sparse matrices in the default dense structure results
in wated memory and processing on the zeros.
To take advantage of the sparse structure of the matrix, the ``CSRNDArray`` in MXNet
stores the matrix in [compressed sparse row(CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) format
and uses specialized algorithms in operators.
The format is designed for 2D matrices with a large number of columns,
and each row is sparse(i.e. with only a few nonzeros).
For matrices of high sparsity (e.g. ~1% non-zeros), the advantage of ``CSRNDArray`` over
the existing ``NDArray`` is that

- memory consumption is reduced significantly
- certain operations (e.g. matrix-vector multiplication) are much faster

Meanwhile, ``CSRNDArray`` inherits competitve features from ``NDArray`` such as
lazy evaluation and automatic parallelization, which is not available in the
scientific computing python package [SciPy](https://www.scipy.org/).

Apart from often queried attributes such as **ndarray.shape**, **ndarray.dtype** and **ndarray.context**,
you’ll also want to query **ndarray.stype**: the storage type of the NDArray. For a usual dense NDArray,
the value of stype is **"default"**. For an CSRNDArray, the value of stype is **"csr"**.

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/get_started/install.html)
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```

## Compressed Sparse Row Format

A CSRNDArray represents a 2D matrix as three separate 1D arrays: **data**,
**indptr** and **indices**, where the column indices for
row ``i`` are stored in ``indices[indptr[i]:indptr[i+1]]`` in ascending order,
and their corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.

For example, the CSR representation for matrix
```
[[7, 0, 8, 0]
 [0, 0, 0, 0]
 [0, 9, 0, 0]]
```
is:
```
[7, 8, 9]          # data
[0, 2, 1]          # indices
[0, 2, 2, 3]       # indptr
```

Note that in MXNet, the column indices for a given row are always sorted in ascending order,
and duplicated column entries for the same row are not allowed.

## Array Creation

There are a few different ways to create a `CSRNDArray`.

* We can create a CSRNDArray with data, indices and indptr by using the `csr_matrix` function:

```python
import mxnet as mx
import numpy as np
# create a CSRNDArray with python lists
shape = (3, 4)
data_list = [7, 8, 9]
indptr_list = [0, 2, 2, 3]
indices_list = [0, 2, 1]
a = mx.nd.sparse.csr_matrix(data_list, indptr_list, indices_list, shape)
# create a CSRNDArray with numpy arrays
data_np = np.array([7, 8, 9])
indptr_np = np.array([0, 2, 2, 3])
indices_np = np.array([0, 2, 1])
b = mx.nd.sparse.csr_matrix(data_np, indptr_np, indices_np, shape)
{'a':a, 'b':b}
```

* We can also create an MXNet CSRNDArray from a `scipy.sparse.csr.csr_matrix` object by using the `array` function:

```python
import scipy.sparse as spsp
# generate a csr matrix in scipy
c = spsp.csr.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
# create a CSRNDArray from a scipy csr object
d = mx.nd.sparse.array(c)
{'d':d}
```

We can specify the element data type with the option `dtype`, which accepts a numpy
type. By default, `float32` is used.

```python
# float32 is used in default
e = mx.nd.sparse.array(a)
# create a 16-bit float array
f = mx.nd.array(a, dtype=np.float16)
(e.dtype, f.dtype)
```

## Inspecting Arrays

* We can inspect the contents of an `CSRNDArray` by filling
its contents into a dense `numpy.ndarray` using the `asnumpy` function.

```python
a.asnumpy()
```

* We can also inspect the internal storage of a CSRNDArray by accessing attributes such as `indptr`, `indices` and `data`:

```python
# access data array
data = a.data
# access indices array
indices = a.indices
# access indptr array
indptr = a.indptr
{'a.stype': a.stype, 'data':data, 'indices':indices, 'indptr':indptr}
```

## Storage Type Conversion

* We can convert an NDArray to a CSRNDArray and vice versa by using the ``tostype`` function:

```python
# create a dense NDArray
ones = mx.nd.ones((2,2))
# cast the storage type from default to csr
csr = ones.tostype('csr')
# cast the storage type from csr to default
dense = csr.tostype('default')
{'csr':csr, 'dense':dense}
```

* We can also convert the storage type by using the ``cast_storage`` operator:

```python
# create a dense NDArray
ones = mx.nd.ones((2,2))
# cast the storage type to csr
csr = mx.nd.sparse.cast_storage(ones, 'csr')
# cast the storage type to default
dense = mx.nd.sparse.cast_storage(csr, 'default')
{'csr':csr, 'dense':dense}
```

## Copies

When assigning an CSRNDArray to another Python variable, we copy a reference to the
*same* CSRNDArray. However, we often need to make a copy of the data, so that we
can manipulate the new array without overwriting the original values.

```python
a = mx.nd.sparse.zeros('csr', (2,2))
b = a
b is a # will be True
```

The `copy` method makes a deep copy of the array and its data:

```python
b = a.copy()
b is a  # will be False
```

The above code allocates a new CSRNDArray and then assigns to *b*. When we do not
want to allocate additional memory, we can use the `copyto` method or the slice
operator `[]` instead.

```python
b = mx.nd.sparse.zeros('csr', a.shape)
c = b
c[:] = a
d = b
a.copyto(d)
(c is b, d is b)  # Both will be True
```

If the storage types of source array and destination array doesn't match,
the storage type of destination array won't change when copying with `copyto` or
the slice operator `[]`.

```python
e = mx.nd.sparse.zeros('csr', (2,2))
f = mx.nd.ones(e.shape)
g = e
g[:] = f
h = e
f.copyto(h)
{'g.stype':g.stype, 'h.stype':h.stype}
```

## Indexing and Slicing

We can slice a CSRNDArray on axis 0 with operator `[]`, which copies the slices and returns a new CSRNDArray.

```python
a = mx.nd.array(np.arange(6).reshape(3,2)).tostype('csr')
b = a[1:2]
c = a[:].asnumpy()
{'b':b, 'c':c}
```

## Sparse Operators and Storage Type Inference

Operators that have specialized implementation for sparse arrays can be accessed in ``mx.nd.sparse``.
You can read the [mxnet.ndarray.sparse API documentation](mxnet.io/api/python/ndarray.html) to find
what sparse operators are available.

For any sparse operator, the storage type of output array is inferred based on inputs. You can either read
the documentation or inspect the `stype` attribute of output array to know what storage type is inferred:

```python
shape = (3, 4)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
a = mx.nd.sparse.csr_matrix(data, indptr, indices, shape)
b = a * 2  # b will be a CSRNDArray since zero multiplied by 2 is still zero
c = a + 1  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```

For operators that don't specialize in sparse arrays, we can still use them with sparse inputs with some performance penalty.
What happens is that MXNet will generate temporary dense inputs from sparse inputs so that the dense operators can be used.
Warning messages will be printed when such storage fallback event happens.

```python
d = mx.nd.log(a) # warnings will be printed
{'a.stype':a.stype, 'd':d} # stype of a is not changed
```

## Loading Sparse Data

Sparse data stored in libsvm file format can be loaded with [mx.io.LibSVMIter](https://mxnet.incubator.apache.org/versions/master/api/python/io.html#mxnet.io.LibSVMIter).
Note that the indices are expected to be zero-based instead of one-based.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
