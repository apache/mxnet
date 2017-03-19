# Data Loading API

## Overview

This document summeries supported data formats and iterator APIs to read the
data including

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.io
    mxnet.recordio
    mxnet.image
```

It will also show how to write an iterator for a new data format.

A data iterator reads data batch by batch.

```python
>>> data = mx.nd.ones((100,10))
>>> nd_iter = mx.io.NDArrayIter(data, batch_size=25)
>>> for batch in nd_iter:
...     print(batch.data)
[<NDArray 25x10 @cpu(0)>]
[<NDArray 25x10 @cpu(0)>]
[<NDArray 25x10 @cpu(0)>]
[<NDArray 25x10 @cpu(0)>]
```

If `nd_iter.reset()` is called, then reads the data again from beginning.

In addition, an iterator provides information about the batch, including the
shapes and name.

```python
>>> nd_iter = mx.io.NDArrayIter(data={'data':mx.nd.ones((100,10))},
...                             label={'softmax_label':mx.nd.ones((100,))},
...                             batch_size=25)
>>> print(nd_iter.provide_data)
[DataDesc[data,(25, 10L),<type 'numpy.float32'>,NCHW]]
>>> print(nd_iter.provide_label)
[DataDesc[softmax_label,(25,),<type 'numpy.float32'>,NCHW]]
```

So this iterator can be used to train a symbol whose input data variable has
name `data` and input label variable has name `softmax_label`.


```python
>>> data = mx.sym.Variable('data')
>>> label = mx.sym.Variable('softmax_label')
>>> fullc = mx.sym.FullyConnected(data=data, num_hidden=1)
>>> loss = mx.sym.SoftmaxOutput(data=data, label=label)
>>> mod = mx.mod.Module(loss)
>>> print(mod.data_names)
['data']
>>> print(mod.label_names)
['softmax_label']
>>> mod.bind(data_shapes=nd_iter.provide_data, label_shapes=nd_iter.provide_label)
```

Then we can call `mod.fit(nd_iter, num_epoch=2)` to train `loss` by 2 epochs.

## Data iterators

```eval_rst
    .. currentmodule:: mxnet
```

```eval_rst
.. autosummary::
    :nosignatures:

    io.NDArrayIter
    io.CSVIter
    io.ImageRecordIter
    io.ImageRecordUInt8Iter
    io.MNISTIter
    recordio.MXRecordIO
    recordio.MXIndexedRecordIO
    image.ImageIter
```

## Helper class

```eval_rst
.. autosummary::
    :nosignatures:

    io.DataDesc
    io.DataBatch
    io.DataIter
    io.ResizeIter
    io.PrefetchingIter
    io.MXDataIter
```

```eval_rst
.. autosummary::
    :nosignatures:

    image.imdecode
    image.scale_down
    image.resize_short
    image.fixed_crop
    image.random_crop
    image.center_crop
    image.color_normalize
    image.random_size_crop
    image.ResizeAug
    image.RandomCropAug
    image.RandomSizedCropAug
    image.CenterCropAug
    image.RandomOrderAug
    image.ColorJitterAug
    image.LightingAug
    image.ColorNormalizeAug
    image.HorizontalFlipAug
    image.CastAug
    image.CreateAugmenter
```

```eval_rst
.. autosummary::
    :nosignatures:

    recordio.pack
    recordio.unpack
    recordio.unpack_img
    recordio.pack_img
```

## Develop a new iterator



## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.io
    :members:
.. automodule:: mxnet.image
    :members:
.. automodule:: mxnet.recordio
    :members:
```
<script>auto_index("api-reference");</script>
