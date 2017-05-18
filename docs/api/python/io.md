# Data Loading API

## Overview

This document summarizes supported data formats and iterator APIs to read the
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

## Helper classes and functions


Data structures and other iterators provided in the ``mxnet.io`` packages.

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

A list of image modification functions provided by ``mxnet.image``.

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

Functions to read and write RecordIO files.

```eval_rst
.. autosummary::
    :nosignatures:

    recordio.pack
    recordio.unpack
    recordio.unpack_img
    recordio.pack_img
```

## Develop a new iterator

Writing a new data iterator in Python is straightforward. Most MXNet
training/inference programs accept an iteratable object with ``provide_data``
and ``provide_label`` properties.
This [tutorial](http://mxnet.io/tutorials/python/data.html#data-iterators) explains how to
write an iterator from scratch.

The following example demonstrates how to combine
multiple data iterators into a single one. It can be used for multiple
modality training such as image captioning, in which images are read by
``ImageRecordIter`` while documents are read by ``CSVIter``

```python
class MultiIter:
    def __init__(self, iter_list):
        self.iters = iter_list
    def next(self):
        batches = [i.next() for i in self.iters]
        return DataBatch(data=[*b.data for b in batches],
                         label=[*b.label for b in batches])
    def reset(self):
        for i in self.iters:
            i.reset()
    @property
    def provide_data(self):
        return [*i.provide_data for i in self.iters]
    @property
    def provide_label(self):
        return [*i.provide_label for i in self.iters]

iter = MultiIter([mx.io.ImageRecordIter('image.rec'), mx.io.CSVIter('txt.csv')])
```

Parsing and performing another pre-processing such as augmentation may be expensive.
If performance is critical, we can implement a data iterator in C++. Refer to
[src/io](https://github.com/dmlc/mxnet/tree/master/src/io) for examples.

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
