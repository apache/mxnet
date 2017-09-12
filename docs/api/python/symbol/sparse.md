# Sparse Symbol API

```eval_rst
    .. currentmodule:: mxnet.symbol.sparse
```

## Overview

This document lists the routines of the sparse symbolic expression package:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.symbol.sparse
```

The `Sparse Symbol` API, defined in the `symbol.sparse` package, provides
sparse neural network graphs and auto-differentiation on CPU.

The storage type of a variable is speficied by the `stype` attribute of the variable.
The storage type of a symbolic expression is inferred based on the storage types of the variables and the operators.

```python
>>> a = mx.sym.Variable('a', stype='csr')
>>> b = mx.sym.Variable('b')
>>> c = mx.sym.dot(a, b, transpose_a=True)
>>> type(c)
<class 'mxnet.symbol.Symbol'>
>>> e = c.bind(mx.cpu(), {'a': mx.nd.array([[1,0,0]]).tostype('csr'), 'b':mx.nd.ones((1,2))})
>>> y = e.forward()
# the result storage type of dot(csr.T, dense) is inferred to be `row_sparse`
>>> y
[<RowSparseNDArray 3x2 @cpu(0)>]
>>> y[0].asnumpy()
array([ 1.,  1.],
      [ 0.,  0.],
      [ 0.,  0.]], dtype=float32)
```

```eval_rst

.. note:: most operators provided in ``mxnet.symbol.sparse`` are similar to those in
   ``mxnet.symbol`` although there are few differences:

   - Only a subset of operators in ``mxnet.symbol`` have specialized implementations in ``mxnet.symbol.sparse``.
     Operators such as reduction and broadcasting do not have sparse implementations yet.
   - The storage types (``stype``) of sparse operators' outputs depend on the storage types of inputs.
     By default the operators not available in ``mxnet.symbol.sparse`` infer "default" (dense) storage type for outputs.
     Please refer to the API reference section for further details on specific operators.
   - GPU support for ``mxnet.symbol.sparse`` is experimental.

```

In the rest of this document, we list sparse related routines provided by the
`symbol.sparse` package.

## Symbol creation routines

```eval_rst
.. autosummary::
    :nosignatures:

    zeros_like
    mxnet.symbol.var
```

## Symbol manipulation routines

### Changing symbol storage type

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

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.symbol.sparse
    :members:

```

<script>auto_index("api-reference");</script>
