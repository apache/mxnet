NDArray API
===========

NDArray package (`mxnet.ndarray`) contains tensor operations similar to `numpy.ndarray`. The syntax is similar except for some additional calls to deal with I/O and multi-devices.

Create NDArray
--------------
Like `numpy`, you could create `mxnet.ndarray` like followings:
```python
>>> import mxnet as mx
>>> # all-zero array of dimension 100x50
>>> a = mx.nd.zeros((100, 50))
>>> # all-one array of dimension 256x32x128x1
>>> b = mx.nd.ones((256, 32, 128, 1))
>>> # initialize array with contents
>>> c = mx.nd.array([[1, 2, 3], [4, 5, 6]])
```

NDArray operations
-------------------
We provide some basic ndarray operations like arithmetic and slice operations. More operations are coming in handy!

### Arithmetic operations
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

### Slice operations
```python
>>> import mxnet as mx
>>> a = mx.nd.zeros((100, 50))
>>> a[0:10] = 1   # first 10 rows will become 1
```

Conversion from/to `numpy.ndarray`
----------------------------------
MXNet NDArray supports pretty nature way to convert from/to `mxnet.ndarray` to/from `numpy.ndarray`:
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

Save Load NDArray
-----------------
You can always use pickle to save and load NDArrays.
We also provide functions to help save and load list or dictionary of NDArrays from file systems.
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
The good thing about using the above `save` and `load` interface is that:
- You could use the format across all `mxnet` language bindings.
- Already support S3 and HDFS.

Multi-device Support
--------------------
The device information is stored in `mxnet.Context` structure. When creating ndarray in mxnet, user could either use the context argument (default is CPU context) to create arrays on specific device or use the `with` statement as follows:
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

Currently, we *DO NOT* allow operations among arrays from different contexts. To allow this, use `copyto` member function to copy the content to different devices and continue computation:
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

NDArray API Reference
---------------------

```eval_rst
.. automodule:: mxnet.ndarray
    :members:

.. raw:: html

    <script>auto_index("mxnet.ndarray");</script>
```

NDArray Random API Reference
----------------------------

```eval_rst
.. automodule:: mxnet.random
    :members:

.. raw:: html

    <script>auto_index("mxnet.random");</script>
```


Context API Reference
---------------------

```eval_rst
.. automodule:: mxnet.context
    :members:

.. raw:: html

    <script>auto_index("mxnet.context");</script>
```
