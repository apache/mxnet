# CSRNDArray - NDArray in Compressed Sparse Row Storage Format

Many real world datasets deal with high dimensional sparse feature vectors. For instance,
in a recommendation system, the number of categories and users is in the order of millions,
while most users only make a few purchases, leading to feature vectors with high sparsity
(i.e. most of the elements are zeros).

Storing and manipulating such large sparse matrices in the default dense structure results
in wasted memory and processing on the zeros.
To take advantage of the sparse structure of the matrix, the ``CSRNDArray`` in MXNet
stores the matrix in [compressed sparse row (CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) format
and uses specialized algorithms in operators.
The format is designed for 2D matrices with a large number of columns,
and each row is sparse (i.e. with only a few nonzeros).
For matrices of high sparsity (e.g. ~1% non-zeros), the advantage of ``CSRNDArray`` over
the existing ``NDArray`` is that

- memory consumption is reduced significantly, and
- certain operations (e.g. matrix-vector multiplication) are much faster.

Meanwhile, ``CSRNDArray`` inherits competitive features from ``NDArray`` such as
lazy evaluation and automatic parallelization, which are not available in the
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
- Scipy - A section of this tutorial uses Scipy package in python. If you don't have Scipy,
the example in that section will be ignored.
- GPUs - A section of this tutorial uses GPUs. If you don't have GPUs on your
machine, simply set the variable gpu_device (set in the GPUs section of this
tutorial) to mx.cpu().

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
# `data` array holds all the non-zero entries of the matrix in row-major order.
data = [7, 8, 9]
# `indices` array stores the column index for each non-zero element in `data`.
indices = [0, 2, 1]
# `indptr` array stores the index into `data` of the first non-zero element number of each row of the matrix.
# The first non-zero entry for row 0 = data[indptr[0]] = 7, with column index = indices[indptr[0]] = 0
# The number of non-zero entries for row 0 = indptr[1] - indptr[0] = 2
# The number of non-zero entries for row 1 = indptr[2] - indptr[1] = 0
# The first non-zero entry for row 2 = data[indptr[2]] = 9, with column index = indices[indptr[2]] = 1
# The number of non-zero entries for row 2 = indptr[3] - indptr[2] = 1
indptr = [0, 2, 2, 3]
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
try:
    import scipy.sparse as spsp
    # generate a csr matrix in scipy
    c = spsp.csr.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
    # create a CSRNDArray from a scipy csr object
    d = mx.nd.sparse.array(c)
    {'d':d}
except ImportError:
    print("scipy package is required")
```

We can specify the element data type with the option `dtype`, which accepts a numpy
type. By default, `float32` is used.

```python
# float32 is used by default
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

* We can use the `copy` method which makes a deep copy of the array and its data, and returns a new array.
We can also use the `copyto` method or the slice operator `[]` to deep copy to an existing array.

```python
a = mx.nd.ones((2,2)).tostype('csr')
b = a.copy()
c = mx.nd.sparse.zeros('csr', (2,2))
c[:] = a
d = mx.nd.sparse.zeros('csr', (2,2))
a.copyto(d)
{'b is a': b is a, 'b.asnumpy()':b.asnumpy(), 'c.asnumpy()':c.asnumpy(), 'd.asnumpy()':d.asnumpy()}
```

* If the storage types of source array and destination array do not match,
the storage type of destination array will not change when copying with `copyto` or
the slice operator `[]`.

```python
e = mx.nd.sparse.zeros('csr', (2,2))
f = mx.nd.sparse.zeros('csr', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype}
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

* Operators that have specialized implementation for sparse arrays can be accessed in ``mx.nd.sparse``.
You can read the [mxnet.ndarray.sparse API documentation](mxnet.io/api/python/ndarray.html) to find
what sparse operators are available.

```python
shape = (3, 4)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
a = mx.nd.sparse.csr_matrix(data, indptr, indices, shape) # a csr matrix as lhs
rhs = mx.nd.ones((4, 1))      # a dense vector as rhs
out = mx.nd.sparse.dot(a, rhs)  # invoke sparse dot operator specialized for dot(csr, dense)
{'out':out}
```

* For any sparse operator, the storage type of output array is inferred based on inputs. You can either read
the documentation or inspect the `stype` attribute of output array to know what storage type is inferred:

```python
b = a * 2  # b will be a CSRNDArray since zero multiplied by 2 is still zero
c = a + 1  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```

* For operators that don't specialize in sparse arrays, we can still use them with sparse inputs with some performance penalty.
In MXNet, dense operators require all inputs and outputs to be in the dense format.
If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily so that the dense operator can be used.
If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format.
Warning messages will be printed when such a storage fallback event happens.

```python
e = mx.nd.sparse.zeros('csr', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd':d, 'e':e} # stypes of a and e will be not changed
```

## Data Loading

* We can load data in batches from a CSRNDArray using ``mx.io.NDArrayIter``:

```python
# create the source CSRNDArray
data = mx.nd.array(np.arange(40).reshape((10,4))).tostype('csr')
labels = np.ones([10, 1])
batch_size = 3
dataiter = mx.io.NDArrayIter(data, labels, batch_size, last_batch_handle='discard')
# inspect the data batches
[batch.data[0] for batch in dataiter]
```

* We can also load data stored in the libsvm file format using ``mx.io.LibSVMIter``:

```python
# create a sample libsvm file in current working directory
import os
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data.t')
with open(data_path, 'w') as fout:
    fout.write('1.0 0:1 2:2\n')
    fout.write('1.0 0:3 5:4\n')
    fout.write('1.0 2:5 8:6 9:7\n')
    fout.write('1.0 3:8\n')
    fout.write('-1 0:0.5 9:1.5\n')
    fout.write('-2.0\n')
    fout.write('-3.0 0:-0.6 1:2.25 2:1.25\n')
    fout.write('-3.0 1:2 2:-1.25\n')
    fout.write('4 2:-1.2\n')

# load CSRNDArrays from the file
data_train = mx.io.LibSVMIter(data_libsvm=data_path, data_shape=(10,), label_shape=(1,), batch_size=3)
for batch in data_train:
    print(data_train.getdata())
    print(data_train.getlabel())

```

Note that in the file the column indices are expected to be sorted in ascending order per row, and be zero-based instead of one-based.

## Advanced Topics

### GPU Support

By default, CSRNDArray operators are executed on CPU. In MXNet, GPU support for CSRNDArray is experimental
with only a few sparse operators such as cast_storage and dot.

To create a CSRNDArray on gpu, we need to explicitly specify the context:

**Note** In order to execute the following section on a cpu set gpu_device to mx.cpu().
```python
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.

a = mx.nd.sparse.zeros('csr', (100, 100), ctx=gpu_device)
a
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
