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

First, let's see how to write an iterator for a new data format.
The following iterator can be used to train a symbol whose input data variable has
name `data` and input label variable has name `softmax_label`.
The iterator also provides information about the batch, including the
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

Let's see a complete example of how to use data iterator in model training.
```python
>>> data = mx.sym.Variable('data')
>>> label = mx.sym.Variable('softmax_label')
>>> fullc = mx.sym.FullyConnected(data=data, num_hidden=1)
>>> loss = mx.sym.SoftmaxOutput(data=fullc, label=label)
>>> mod = mx.mod.Module(loss, data_names=['data'], label_names=['softmax_label'])
>>> mod.bind(data_shapes=nd_iter.provide_data, label_shapes=nd_iter.provide_label)
>>> mod.fit(nd_iter, num_epoch=2)
```

A detailed tutorial is available at
[Iterators - Loading data](http://mxnet.io/tutorials/basic/data.html).

## Data iterators

```eval_rst
    .. currentmodule:: mxnet
```

```eval_rst
.. autosummary::
    :nosignatures:

    io.NDArrayIter
    io.CSVIter
    io.LibSVMIter
    io.ImageRecordIter
    io.ImageRecordUInt8Iter
    io.MNISTIter
    recordio.MXRecordIO
    recordio.MXIndexedRecordIO
    image.ImageIter
    image.ImageDetIter
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
training/inference programs accept an iterable object with ``provide_data``
and ``provide_label`` properties.
This [tutorial](http://mxnet.io/tutorials/basic/data.html) explains how to
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

### Change batch layout

By default, the backend engine treats the first dimension of each data and label variable in data
iterators as the batch size (i.e. `NCHW` or `NT` layout). In order to override the axis for batch size,
the `provide_data` (and `provide_label` if there is label) properties should include the layouts. This
is especially useful in RNN since `TNC` layouts are often more efficient. For example:

```python
@property
def provide_data(self):
    return [DataDesc(name='seq_var', shape=(seq_length, batch_size), layout='TN')]
```
The backend engine will recognize the index of `N` in the `layout` as the axis for batch size.

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.io
    :members:
.. automodule:: mxnet.recordio
    :members:
```
<script>auto_index("api-reference");</script>
