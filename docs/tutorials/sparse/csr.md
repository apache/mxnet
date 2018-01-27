
# CSRNDArray - NDArray in Compressed Sparse Row Storage Format

Many real world datasets deal with high dimensional sparse feature vectors. Take for instance a recommendation system where the number of categories and users is on the order of millions. The purchase data for each category by user would show that most users only make a few purchases, leading to a dataset with high sparsity (i.e. most of the elements are zeros).

Storing and manipulating such large sparse matrices in the default dense structure results in wasted memory and processing on the zeros. To take advantage of the sparse structure of the matrix, the `CSRNDArray` in MXNet stores the matrix in [compressed sparse row (CSR)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) format and uses specialized algorithms in operators.
**The format is designed for 2D matrices with a large number of columns,
and each row is sparse (i.e. with only a few nonzeros).**

## Advantages of Compressed Sparse Row NDArray (CSRNDArray)
For matrices of high sparsity (e.g. ~1% non-zeros = ~1% density), there are two primary advantages of `CSRNDArray` over the existing `NDArray`:

- memory consumption is reduced significantly
- certain operations are much faster (e.g. matrix-vector multiplication)

You may be familiar with the CSR storage format in [SciPy](https://www.scipy.org/) and will note the similarities in MXNet's implementation. However there are some additional competitive features in `CSRNDArray` inherited from `NDArray`, such as non-blocking asynchronous evaluation and automatic parallelization that are not available in SciPy's flavor of CSR. You can find further explainations for evaluation and parallization strategy in MXNet in the [NDArray tutorial](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html#lazy-evaluation-and-automatic-parallelization).

The introduction of `CSRNDArray` also brings a new attribute, `stype` as a holder for storage type info, to `NDArray`. You can query **ndarray.stype** now in addition to the oft-queried attributes such as **ndarray.shape**, **ndarray.dtype**, and **ndarray.context**. For a typical dense NDArray, the value of `stype` is **"default"**. For a `CSRNDArray`, the value of stype is **"csr"**.

## Prerequisites

To complete this tutorial, you will need:

- MXNet. See the instructions for your operating system in [Setup and Installation](https://mxnet.io/install/index.html)
- [Jupyter](http://jupyter.org/)
    ```
    pip install jupyter
    ```
- Basic knowledge of NDArray in MXNet. See the detailed tutorial for NDArray in [NDArray - Imperative tensor operations on CPU/GPU](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html).
- SciPy - A section of this tutorial uses SciPy package in Python. If you don't have SciPy, the example in that section will be ignored.
- GPUs - A section of this tutorial uses GPUs. If you don't have GPUs on your machine, simply set the variable `gpu_device` (set in the GPUs section of this tutorial) to `mx.cpu()`.

## Compressed Sparse Row Matrix

A CSRNDArray represents a 2D matrix as three separate 1D arrays: **data**, **indptr** and **indices**, where the column indices for row `i` are stored in `indices[indptr[i]:indptr[i+1]]` in ascending order, and their corresponding values are stored in `data[indptr[i]:indptr[i+1]]`.

- **data**: CSR format data array of the matrix
- **indices**: CSR format index array of the matrix
- **indptr**: CSR format index pointer array of the matrix

### Example Matrix Compression

For example, given the matrix:
```
[[7, 0, 8, 0]
 [0, 0, 0, 0]
 [0, 9, 0, 0]]
```

We can compress this matrix using CSR, and to do so we need to calculate `data`, `indices`, and `indptr`.

The `data` array holds all the non-zero entries of the matrix in row-major order. Put another way, you create a data array that has all of the zeros removed from the matrix, row by row, storing the numbers in that order. Your result:

    data = [7, 8, 9]

The `indices` array stores the column index for each non-zero element in `data`. As you cycle through the data array, starting with 7, you can see it is in column 0. Then looking at 8, you can see it is in column 2. Lastly 9 is in column 1. Your result:

    indices = [0, 2, 1]

The `indptr` array is what will help identify the rows where the data appears. It stores the offset into `data` of the first non-zero element number of each row of the matrix. This array always starts with 0 (reasons can be explored later), so indptr[0] is 0. Each subsequent value in the array is the aggregate number of non-zero elements up to that row. Looking at the first row of the matrix you can see two non-zero values, so indptr[1] is 2. The next row contains all zeros, so the aggregate is still 2, so indptr[2] is 2. Finally, you see the last row contains one non-zero element bring the aggregate to 3, so indptr[3] is 3. To reconstruct the dense matrix, you will use `data[0:2]` and `indices[0:2]` for the first row, `data[2:2]` and `indices[2:2]` for the second row (which contains all zeros), and `data[2:3]` and `indices[2:3]` for the third row. Your result:

    indptr = [0, 2, 2, 3]

Note that in MXNet, the column indices for a given row are always sorted in ascending order,
and duplicated column indices for the same row are not allowed.

## Array Creation

There are a few different ways to create a `CSRNDArray`, but first let's recreate the matrix we just discussed using the `data`, `indices`, and `indptr` we calculated in the previous example.

You can create a CSRNDArray with data, indices and indptr by using the `csr_matrix` function:


```python
import mxnet as mx
# Create a CSRNDArray with python lists
shape = (3, 4)
data_list = [7, 8, 9]
indices_list = [0, 2, 1]
indptr_list = [0, 2, 2, 3]
a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
# Inspect the matrix
a.asnumpy()
```




    array([[ 7.,  0.,  8.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  9.,  0.,  0.]], dtype=float32)




```python
import numpy as np
# Create a CSRNDArray with numpy arrays
data_np = np.array([7, 8, 9])
indptr_np = np.array([0, 2, 2, 3])
indices_np = np.array([0, 2, 1])
b = mx.nd.sparse.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
b.asnumpy()
```




    array([[7, 0, 8, 0],
           [0, 0, 0, 0],
           [0, 9, 0, 0]])




```python
# Compare the two. They are exactly the same.
{'a':a.asnumpy(), 'b':b.asnumpy()}
```




    {'a': array([[ 7.,  0.,  8.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  9.,  0.,  0.]], dtype=float32), 'b': array([[7, 0, 8, 0],
            [0, 0, 0, 0],
            [0, 9, 0, 0]])}



You can create an MXNet CSRNDArray from a `scipy.sparse.csr.csr_matrix` object by using the `array` function:


```python
try:
    import scipy.sparse as spsp
    # generate a csr matrix in scipy
    c = spsp.csr.csr_matrix((data_np, indices_np, indptr_np), shape=shape)
    # create a CSRNDArray from a scipy csr object
    d = mx.nd.sparse.array(c)
    print('d:{}'.format(d.asnumpy()))
except ImportError:
    print("scipy package is required")
```

    d:[[7 0 8 0]
     [0 0 0 0]
     [0 9 0 0]]


What if you have a big set of data and you haven't calculated indices or indptr yet? Let's try a simple CSRNDArray from an existing array of data and derive those values with some built-in functions. We can mockup a "big" dataset with a random amount of the data being non-zero, then compress it by using the `tostype` function, which is explained further in the [Storage Type Conversion](#storage-type-conversion) section:


```python
big_array = mx.nd.round(mx.nd.random.uniform(low=0, high=1, shape=(1000, 100)))
print(big_array)
big_array_csr = big_array.tostype('csr')
# Access indices array
indices = big_array_csr.indices
# Access indptr array
indptr = big_array_csr.indptr
# Access data array
data = big_array_csr.data
# The total size of `data`, `indices` and `indptr` arrays is much lesser than the dense big_array!
```

    
    [[ 1.  1.  0. ...,  0.  1.  1.]
     [ 0.  0.  0. ...,  0.  0.  1.]
     [ 1.  0.  0. ...,  1.  0.  0.]
     ..., 
     [ 0.  1.  1. ...,  0.  0.  0.]
     [ 1.  1.  0. ...,  1.  0.  1.]
     [ 1.  0.  1. ...,  1.  0.  0.]]
    <NDArray 1000x100 @cpu(0)>


You can also create a CSRNDArray from another using the `array` function specifying the element data type with the option `dtype`,
which accepts a numpy type. By default, `float32` is used.


```python
# Float32 is used by default
e = mx.nd.sparse.array(a)
# Create a 16-bit float array
f = mx.nd.array(a, dtype=np.float16)
(e.dtype, f.dtype)
```




    (numpy.float32, numpy.float16)



## Inspecting Arrays

A variety of methods are available for you to use for inspecting CSR arrays:
* **.asnumpy()**
* **.data**
* **.indices**
* **.indptr**

As you have seen already, we can inspect the contents of a `CSRNDArray` by filling
its contents into a dense `numpy.ndarray` using the `asnumpy` function.


```python
a.asnumpy()
```




    array([[ 7.,  0.,  8.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  9.,  0.,  0.]], dtype=float32)



You can also inspect the internal storage of a CSRNDArray by accessing attributes such as `indptr`, `indices` and `data`:


```python
# Access data array
data = a.data
# Access indices array
indices = a.indices
# Access indptr array
indptr = a.indptr
{'a.stype': a.stype, 'data':data, 'indices':indices, 'indptr':indptr}
```




    {'a.stype': 'csr', 'data': 
     [ 7.  8.  9.]
     <NDArray 3 @cpu(0)>, 'indices': 
     [0 2 1]
     <NDArray 3 @cpu(0)>, 'indptr': 
     [0 2 2 3]
     <NDArray 4 @cpu(0)>}



## Storage Type Conversion

You can also convert storage types with:
* **tostype**
* **cast_storage**

To convert an NDArray to a CSRNDArray and vice versa by using the ``tostype`` function:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type from `default` to `csr`
csr = ones.tostype('csr')
# Cast the storage type from `csr` to `default`
dense = csr.tostype('default')
{'csr':csr, 'dense':dense}
```




    {'csr': 
     <CSRNDArray 2x2 @cpu(0)>, 'dense': 
     [[ 1.  1.]
      [ 1.  1.]]
     <NDArray 2x2 @cpu(0)>}



To convert the storage type by using the `cast_storage` operator:


```python
# Create a dense NDArray
ones = mx.nd.ones((2,2))
# Cast the storage type to `csr`
csr = mx.nd.sparse.cast_storage(ones, 'csr')
# Cast the storage type to `default`
dense = mx.nd.sparse.cast_storage(csr, 'default')
{'csr':csr, 'dense':dense}
```




    {'csr': 
     <CSRNDArray 2x2 @cpu(0)>, 'dense': 
     [[ 1.  1.]
      [ 1.  1.]]
     <NDArray 2x2 @cpu(0)>}



## Copies

You can use the `copy` method which makes a deep copy of the array and its data, and returns a new array.
You can also use the `copyto` method or the slice operator `[]` to deep copy to an existing array.


```python
a = mx.nd.ones((2,2)).tostype('csr')
b = a.copy()
c = mx.nd.sparse.zeros('csr', (2,2))
c[:] = a
d = mx.nd.sparse.zeros('csr', (2,2))
a.copyto(d)
{'b is a': b is a, 'b.asnumpy()':b.asnumpy(), 'c.asnumpy()':c.asnumpy(), 'd.asnumpy()':d.asnumpy()}
```




    {'b is a': False, 'b.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32), 'c.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32), 'd.asnumpy()': array([[ 1.,  1.],
            [ 1.,  1.]], dtype=float32)}



If the storage types of source array and destination array do not match,
the storage type of destination array will not change when copying with `copyto` or
the slice operator `[]`.


```python
e = mx.nd.sparse.zeros('csr', (2,2))
f = mx.nd.sparse.zeros('csr', (2,2))
g = mx.nd.ones(e.shape)
e[:] = g
g.copyto(f)
{'e.stype':e.stype, 'f.stype':f.stype, 'g.stype':g.stype}
```




    {'e.stype': 'csr', 'f.stype': 'csr', 'g.stype': 'default'}



## Indexing and Slicing
You can slice a CSRNDArray on axis 0 with operator `[]`, which copies the slices and returns a new CSRNDArray.


```python
a = mx.nd.array(np.arange(6).reshape(3,2)).tostype('csr')
b = a[1:2].asnumpy()
c = a[:].asnumpy()
{'a':a, 'b':b, 'c':c}
```




    {'a': 
     <CSRNDArray 3x2 @cpu(0)>,
     'b': array([[ 2.,  3.]], dtype=float32),
     'c': array([[ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.]], dtype=float32)}



Note that multi-dimensional indexing or slicing along a particular axis is currently not supported for a CSRNDArray.

## Sparse Operators and Storage Type Inference

Operators that have specialized implementation for sparse arrays can be accessed in `mx.nd.sparse`. You can read the [mxnet.ndarray.sparse API documentation](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/sparse.html) to find what sparse operators are available.


```python
shape = (3, 4)
data = [7, 8, 9]
indptr = [0, 2, 2, 3]
indices = [0, 2, 1]
a = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape) # a csr matrix as lhs
rhs = mx.nd.ones((4, 1))      # a dense vector as rhs
out = mx.nd.sparse.dot(a, rhs)  # invoke sparse dot operator specialized for dot(csr, dense)
{'out':out}
```




    {'out': 
     [[ 15.]
      [  0.]
      [  9.]]
     <NDArray 3x1 @cpu(0)>}



For any sparse operator, the storage type of output array is inferred based on inputs. You can either read the documentation or inspect the `stype` attribute of the output array to know what storage type is inferred:


```python
b = a * 2  # b will be a CSRNDArray since zero multiplied by 2 is still zero
c = a + mx.nd.ones(shape=(3, 4))  # c will be a dense NDArray
{'b.stype':b.stype, 'c.stype':c.stype}
```




    {'b.stype': 'csr', 'c.stype': 'default'}



For operators that don't specialize in sparse arrays, we can still use them with sparse inputs with some performance penalty. In MXNet, dense operators require all inputs and outputs to be in the dense format.

If sparse inputs are provided, MXNet will convert sparse inputs into dense ones temporarily, so that the dense operator can be used.

If sparse outputs are provided, MXNet will convert the dense outputs generated by the dense operator into the provided sparse format.


```python
e = mx.nd.sparse.zeros('csr', a.shape)
d = mx.nd.log(a) # dense operator with a sparse input
e = mx.nd.log(a, out=e) # dense operator with a sparse output
{'a.stype':a.stype, 'd.stype':d.stype, 'e.stype':e.stype} # stypes of a and e will be not changed
```




    {'a.stype': 'csr', 'd.stype': 'default', 'e.stype': 'csr'}



Note that warning messages will be printed when such a storage fallback event happens. If you are using jupyter notebook, the warning message will be printed in your terminal console.

## Data Loading

You can load data in batches from a CSRNDArray using `mx.io.NDArrayIter`:


```python
# Create the source CSRNDArray
data = mx.nd.array(np.arange(36).reshape((9,4))).tostype('csr')
labels = np.ones([9, 1])
batch_size = 3
dataiter = mx.io.NDArrayIter(data, labels, batch_size, last_batch_handle='discard')
# Inspect the data batches
[batch.data[0] for batch in dataiter]
```




    [
     <CSRNDArray 3x4 @cpu(0)>, 
     <CSRNDArray 3x4 @cpu(0)>, 
     <CSRNDArray 3x4 @cpu(0)>]



You can also load data stored in the [libsvm file format](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) using `mx.io.LibSVMIter`, where the format is: ``<label> <col_idx1>:<value1> <col_idx2>:<value2> ... <col_idxN>:<valueN>``. Each line in the file records the label and the column indices and data for non-zero entries. For example, for a matrix with 6 columns, ``1 2:1.5 4:-3.5`` means the label is ``1``, the data is ``[[0, 0, 1,5, 0, -3.5, 0]]``. More detailed examples of `mx.io.LibSVMIter` are available in the [API documentation](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter).


```python
# Create a sample libsvm file in current working directory
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

# Load CSRNDArrays from the file
data_train = mx.io.LibSVMIter(data_libsvm=data_path, data_shape=(10,), label_shape=(1,), batch_size=3)
for batch in data_train:
    print(data_train.getdata())
    print(data_train.getlabel())
```

    
    <CSRNDArray 3x10 @cpu(0)>
    
    [ 1.  1.  1.]
    <NDArray 3 @cpu(0)>
    
    <CSRNDArray 3x10 @cpu(0)>
    
    [ 1. -1. -2.]
    <NDArray 3 @cpu(0)>
    
    <CSRNDArray 3x10 @cpu(0)>
    
    [-3. -3.  4.]
    <NDArray 3 @cpu(0)>


Note that in the file the column indices are expected to be sorted in ascending order per row, and be zero-based instead of one-based.

## Advanced Topics

### GPU Support

By default, `CSRNDArray` operators are executed on CPU. In MXNet, GPU support for `CSRNDArray` is experimental with only a few sparse operators such as `cast_storage` and `dot`.

To create a `CSRNDArray` on a GPU, we need to explicitly specify the context:

**Note** If a GPU is not available, an error will be reported in the following section. In order to execute it a cpu, set `gpu_device` to `mx.cpu()`.


```python
import sys
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
try:
    a = mx.nd.sparse.zeros('csr', (100, 100), ctx=gpu_device)
    a
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->



