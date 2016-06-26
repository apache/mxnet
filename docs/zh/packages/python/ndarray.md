NDArray API
===========

NDArray 程序包 (`mxnet.ndarray`) 包含类似于 `numpy.ndarray` 的 张量计算包.  它的语法很相近, 除了增加了一些处理 I/O 和多设备的调用.

Create NDArray
--------------

类似 `numpy`, 你可以按照下面的方式来创建 `mxnet.ndarray` :
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

我们提供了几个基本的 ndarray 操作, 比如说算术和切片. 更多的操作正在开发中!

### 算术操作
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

### 切片操作
```python
>>> import mxnet as mx
>>> a = mx.nd.zeros((100, 50))
>>> a[0:10] = 1   # first 10 rows will become 1
```

Conversion from/to `numpy.ndarray`
----------------------------------

MXNet NDArray 提供了很自然的方式来支持`mxnet.ndarray` 和 `numpy.ndarray` 之间的互相转换:

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

你可以一种使用 pickle 来保存和加载 NDArray.
我们也提供了一些函数来简化 NDArray 的列表或者字典的加载与保存操作.

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

使用 `save` 和 `load` 的好的一方面是:
- 你可以在所有的 `mxnet` 的其他编程语言的绑定中相同的接口.
- 已经支持 S3 和 HDFS

Multi-device Support
--------------------
设备信息是存储在 `mxnet.Context` 数据结构中. 当我们在 mxnet 中创建 ndarray 的时候, 我们要么使用上下文参数(默认是 CPU 上下文) 在指定的设备上创建, 或者按照下面的例子中的方式使用 `with` 表达式:

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

现在我们还 *不支持* 涉及不同上下文环境中的多个 ndarray 的操作. 为了支持这种情况下的操作, 我们首先使用 `copyto` 方法将不同的上下文环境中的 ndarray 拷贝到同一个上下文环境中, 然后执行相应的操作:

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
