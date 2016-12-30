# NDArray API


The NDArray package (`mxnet.ndarray`) contains tensor operations similar to `numpy.ndarray`. The syntax is also similar, except for some additional calls for dealing with I/O and multiple devices.

## Create NDArray

Create `mxnet.ndarray` as follows:

```python
    >>> import mxnet as mx
    >>> # all-zero array of dimension 100x50
    >>> a = mx.nd.zeros((100, 50))
    >>> # all-one array of dimension 256x32x128x1
    >>> b = mx.nd.ones((256, 32, 128, 1))
    >>> # initialize array with contents
    >>> c = mx.nd.array([[1, 2, 3], [4, 5, 6]])
```
This is similar to the way you use `numpy`.
## NDArray Operations

We provide some basic ndarray operations, like arithmetic and slice operations. 

### Arithmetic Operations

```python
    >>> import mxnet as mx
    >>> a = mx.nd.zeros((100, 50))
    >>> a.shape
    (100L, 50L)
    >>> b = mx.nd.ones((100, 50))
    >>> # c and d will be calculated in parallel here!
    >>> c = a + b
    >>> d = a - b
    >>> # inplace operation, b's contents will be modified, but c and d won't be affected.
    >>> b += d
```

### Slice Operations

```python
    >>> import mxnet as mx
    >>> a = mx.nd.zeros((100, 50))
    >>> a[0:10] = 1   # first 10 rows will become 1
```

### Convert from or to numpy.ndarray

MXNet NDArray provides an easy way to convert from or to `mxnet.ndarray` to or from `numpy.ndarray`:

```python
    >>> import mxnet as mx
    >>> import numpy as np
    >>> a = np.array([1,2,3])
    >>> b = mx.nd.array(a)                  # convert from numpy array
    >>> b
    <mxnet.ndarray.NDArray object at ...>
    >>> b.asnumpy()                         # convert to numpy array
    array([ 1., 2., 3.], dtype=float32)
```

### Save and Load NDArray

You can use pickle to save and load NDArrays.
Or, you can use MXNet functions to save and load a list or dictionary of NDArrays from file systems.

```python
    >>> import mxnet as mx
    >>> a = mx.nd.zeros((100, 200))
    >>> b = mx.nd.zeros((100, 200))
    >>> # save list of NDArrays
    >>> mx.nd.save("/path/to/array/file", [a, b])
    >>> # save dictionary of NDArrays to AWS S3
    >>> mx.nd.save("s3://path/to/s3/array", {'A' : a, 'B' : b})
    >>> # save list of NDArrays to hdfs.
    >>> mx.nd.save("hdfs://path/to/hdfs/array", [a, b])
    >>> from_file = mx.nd.load("/path/to/array/file")
    >>> from_s3 = mx.nd.load("s3://path/to/s3/array")
    >>> from_hdfs = mx.nd.load("hdfs://path/to/hdfs/array")
```
The good thing about using the `save` and `load` interface is that you can use the format across all `mxnet` language bindings. They also already support Amazon S3 and HDFS.

### Multi-Device Support

Device information is stored in the `mxnet.Context` structure. When creating NDArray in MXNet, you can use either the context argument (the default is the CPU context) to create arrays on specific devices or the `with` statement, as follows:

```python
    >>> import mxnet as mx
    >>> cpu_a = mx.nd.zeros((100, 200))
    >>> cpu_a.context
    cpu(0)
    >>> with mx.Context(mx.gpu(0)):
    >>>   gpu_a = mx.nd.ones((100, 200))
    >>> gpu_a.context
    gpu(0)
    >>> ctx = mx.Context(mx.gpu(0))
    >>> gpu_b = mx.nd.zeros((100, 200), ctx)
    >>> gpu_b.context
    gpu(0)
```

Currently, we *do not* allow operations among arrays from different contexts. To manually enable this, use the `copyto` member function to copy the content to different devices, and continue computation:

```python
    >>> import mxnet as mx
    >>> x = mx.nd.zeros((100, 200))
    >>> with mx.Context(mx.gpu(0)):
    >>>   y = mx.nd.zeros((100, 200))
    >>> z = x + y
    mxnet.base.MXNetError: [13:29:12] src/ndarray/ndarray.cc:33: Check failed: lhs.ctx() == rhs.ctx() operands context mismatch
    >>> cpu_y = mx.nd.zeros((100, 200))
    >>> y.copyto(cpu_y)
    >>> z = x + cpu_y
```

```eval_rst
    .. raw:: html

        <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```

## NDArray API Reference


```eval_rst
    .. automodule:: mxnet.ndarray
        :members:

    .. raw:: html

        <script>auto_index("ndarray-api-reference");</script>
```

## NDArray Random API Reference


```eval_rst
    .. automodule:: mxnet.random
        :members:

    .. raw:: html

        <script>auto_index("ndarray-random-api-reference");</script>
```


## Context API Reference


```eval_rst
    .. automodule:: mxnet.context
        :members:

    .. raw:: html

        <script>auto_index("context-api-reference");</script>
```

## Next Steps
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
