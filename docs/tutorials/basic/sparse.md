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
the value of stype is "default". For an CSRNDArray, the value of stype is "csr".

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/get_started/install.html)
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
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

## Printing Arrays

We can also inspect the contents of an `CSRNDArray` by filling
its contents to a dense `numpy.ndarray` using the `asnumpy` function.

```python
e.asnumpy()
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

