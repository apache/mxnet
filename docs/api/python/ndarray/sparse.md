# Sparse NDArray API

```eval_rst
.. currentmodule:: mxnet.ndarray.sparse
```

## Overview

This document lists the routines of the *n*-dimensional sparse array package:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.ndarray.sparse
```

The `CSRNDArray` and `RowSparseNDArray` API, defined in the `ndarray.sparse` package, provides
imperative sparse tensor operations on CPU.

An `CSRNDArray` represents a two-dimensional, fixed-size array in compressed sparse row format.

```python
>>> x = mx.nd.array([[1, 0], [0, 0], [2, 3]])
>>> csr = x.tostype('csr')
>>> type(csr)
<class 'mxnet.ndarray.sparse.CSRNDArray'>
>>> csr.shape
(3, 2)
>>> csr.data.asnumpy()
array([ 1.  2.  3.], dtype=float32)
>>> csr.indices.asnumpy()
array([0, 0, 1])
>>> csr.indptr.asnumpy()
array([0, 1, 1, 3])
```

An `RowSparseNDArray` represents a multi-dimensional, fixed-size array in row sparse format.

```python
>>> x = mx.nd.array([[1, 0], [0, 0], [2, 3]])
>>> row_sparse = x.tostype('row_sparse')
>>> type(row_sparse)
<class 'mxnet.ndarray.sparse.RowSparseNDArray'>
>>> row_sparse.data.asnumpy()
array([[ 1.  0.],
       [ 2.  3.]], dtype=float32)
>>> row_sparse.indices.asnumpy()
array([0, 2])
```

<br><br>

```eval_rst

.. note:: ``mxnet.ndarray`` is similar to ``numpy.ndarray`` in some aspects. But the differences are not negligible. For instance:

   - ``mxnet.ndarray.NDArray.T`` does real data transpose to return new a copied 
     array, instead of returning a view of the input array.
   - ``mxnet.ndarray.dot`` performs dot product between the last axis of the
     first input array and the first axis of the second input, while `numpy.dot`
     uses the second last axis of the input array.

   In addition, ``mxnet.ndarray.NDArray`` supports GPU computation and various neural
   network layers.

.. note:: ``ndarray`` provides almost the same routines as ``symbol``. Most
  routines between these two packages share the source code. But ``ndarray``
  differs from ``symbol`` in few aspects:

  - ``ndarray`` adopts imperative programming, namely sentences are executed
    step-by-step so that the results can be obtained immediately whereas 
    ``symbol`` adopts declarative programming.

  - Most binary operators in ``ndarray`` such as ``+`` and ``>`` have
    broadcasting enabled by default.
```

In the rest of this document, we first overview the methods provided by the
`ndarray.sparse.CSRNDArray` class and the `ndarray.sparse.RowSparseNDArray` class,
and then list other routines provided by the `ndarray.sparse` package.

The `ndarray.sparse` package provides several classes:

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray
    RowSparseNDArray
```

We summarize the interface for each class in the following sections.

## The `CSRNDArray` class

### Array attributes

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.shape
    CSRNDArray.size
    CSRNDArray.context
    CSRNDArray.dtype
    CSRNDArray.stype
    CSRNDArray.data
    CSRNDArray.indices
    CSRNDArray.indptr
```

### Array conversion

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.copy
    CSRNDArray.copyto
    CSRNDArray.as_in_context
    CSRNDArray.asnumpy
    CSRNDArray.asscalar
    CSRNDArray.astype
    CSRNDArray.tostype
```

### Array creation

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.zeros_like
```

### Indexing

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.__getitem__
    CSRNDArray.__setitem__
    CSRNDArray.slice
```

### Lazy evaluation

```eval_rst
.. autosummary::
    :nosignatures:

    CSRNDArray.wait_to_read
```

## The `RowSparseNDArray` class

### Array attributes

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.shape
    RowSparseNDArray.size
    RowSparseNDArray.context
    RowSparseNDArray.dtype
    RowSparseNDArray.stype
    RowSparseNDArray.data
    RowSparseNDArray.indices
```

### Array conversion

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.copy
    RowSparseNDArray.copyto
    RowSparseNDArray.as_in_context
    RowSparseNDArray.asnumpy
    RowSparseNDArray.asscalar
    RowSparseNDArray.astype
    RowSparseNDArray.tostype
```

### Array creation

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.zeros_like
```

### Indexing

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.__getitem__
    RowSparseNDArray.__setitem__
```

### Lazy evaluation

```eval_rst
.. autosummary::
    :nosignatures:

    RowSparseNDArray.wait_to_read
```

## Array creation routines

```eval_rst
.. autosummary::
    :nosignatures:

    array
    empty
    zeros
    zeros_like
    csr_matrix
    row_sparse_array
```

## Array manipulation routines

### Changing array storage type

```eval_rst
.. autosummary::
    :nosignatures:

    cast_storage
```

### Indexing routines

```eval_rst
.. autosummary::
    :nosignatures:

    slice
    retain
```

## Mathematical functions

### Arithmetic operations

```eval_rst
.. autosummary::
    :nosignatures:

    elemwise_add
    dot
    add_n
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst

.. autoclass:: mxnet.ndarray.sparse.CSRNDArray
    :members:
    :special-members:

.. autoclass:: mxnet.ndarray.sparse.RowSparseNDArray
    :members:
    :special-members:

.. automodule:: mxnet.ndarray.sparse
    :members: array, empty, zeros
    :special-members:
    :exclude-members: BaseSparseNDArray, RowSparseNDArray, CSRNDArray

```

<script>auto_index("api-reference");</script>
