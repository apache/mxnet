# Record IO - Pack free-format data in binary files

This tutorial will walk through the python interface for reading and writing
record io files. It can be useful when you need more more control over the
details of data pipeline. For example, when you need to augument image and label
together for detection and segmentation, or when you need a custom data iterator
for triplet sampling and negative sampling.

Setup environment first:

```python
%matplotlib inline
from __future__ import print_function
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
```

The relevent code is under `mx.recordio`. There are two classes: `MXRecordIO`,
which supports sequential read and write, and `MXIndexedRecordIO`, which
supports random read and sequential write.

## MXRecordIO

First let's take a look at `MXRecordIO`. We open a file `tmp.rec` and write 5
strings to it:

```python
record = mx.recordio.MXRecordIO('tmp.rec', 'w')
for i in range(5):
    record.write('record_%d'%i)
record.close()
```

Then we can read it back by opening the same file with 'r':

```python
record = mx.recordio.MXRecordIO('tmp.rec', 'r')
while True:
    item = record.read()
    if not item:
        break
    print item
record.close()
```

## MXIndexedRecordIO

Some times you need random access for more complex tasks. `MXIndexedRecordIO` is
designed for this. Here we create a indexed record `tmp.rec` and a corresponding
index file `tmp.idx`:


```python
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
for i in range(5):
    record.write_idx(i, 'record_%d'%i)
record.close()
```

We can then access records with keys:

```python
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
record.read_idx(3)
```

You can list all keys with:

```python
record.keys
```

## Packing and Unpacking Data

Each record in a .rec file can contain arbitrary binary data, but machine
learning data typically has a label/data structure. `mx.recordio` also contains
a few utility functions for packing such data, namely: `pack`, `unpack`,
`pack_img`, and `unpack_img`.

### Binary Data

`pack` and `unpack` are used for storing float (or 1d array of float) label and
binary data:

- pack:

```python
# pack
data = 'data'
label1 = 1.0
header1 = mx.recordio.IRHeader(flag=0, label=label1, id=1, id2=0)
s1 = mx.recordio.pack(header1, data)
print('float label:', repr(s1))
label2 = [1.0, 2.0, 3.0]
header2 = mx.recordio.IRHeader(flag=0, label=label2, id=2, id2=0)
s2 = mx.recordio.pack(header2, data)
print('array label:', repr(s2))
```

- unpack:

```python
print(*mx.recordio.unpack(s1))
print(*mx.recordio.unpack(s2))
```

### Image Data

`pack_img` and `unpack_img` are used for packing image data. Records packed by
`pack_img` can be loaded by `mx.io.ImageRecordIter`.

- pack images

```python
data = np.ones((3,3,1), dtype=np.uint8)
label = 1.0
header = mx.recordio.IRHeader(flag=0, label=label, id=0, id2=0)
s = mx.recordio.pack_img(header, data, quality=100, img_fmt='.jpg')
print(repr(s))
```

- unpack images

```python
print(*mx.recordio.unpack_img(s))
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
