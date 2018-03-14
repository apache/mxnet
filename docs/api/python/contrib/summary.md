# Logging MXNet Data for Visualization in TensorBoard

## Overview

The module `mxnet.contrib.summary` enables MXNet to log data that can be visualized in
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard). 

Please note that this module only provides the APIs for data logging. To launch TensorBoard
for visualization, make sure you have the
[official release of TensorBoard](https://pypi.python.org/pypi/tensorboard) installed.
Then type in the terminal:
```
tensorborad --logdir=/path/to/your/log/dir --host=your_host_ip --port=your_port_number
```
open the browser and enter the address `your_host_ip:your_port_number`. The logged data
will be rendered in the browser when the logger flushes them to the event files.

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
```

The `summary` module provides the logging APIs through the `SummaryWriter` class.

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.contrib.summary.SummaryWriter
    mxnet.contrib.summary.SummaryWriter.add_audio
    mxnet.contrib.summary.SummaryWriter.add_embedding
    mxnet.contrib.summary.SummaryWriter.add_graph
    mxnet.contrib.summary.SummaryWriter.add_histogram
    mxnet.contrib.summary.SummaryWriter.add_image
    mxnet.contrib.summary.SummaryWriter.add_pr_curve
    mxnet.contrib.summary.SummaryWriter.add_scalar
    mxnet.contrib.summary.SummaryWriter.add_text
    mxnet.contrib.summary.SummaryWriter.close
    mxnet.contrib.summary.SummaryWriter.flush
    mxnet.contrib.summary.SummaryWriter.get_logdir
    mxnet.contrib.summary.SummaryWriter.reopen
```

## Examples
Let's take a look at several simple examples showing how to use the logggin APIs.

### Scalar
Scalar values are often plotted in terms of curves, such as training accuracy as time evolves. Here
is an example of plotting the curve of `y=sin(x/100)` where `x` is in the range of `[0, 2*pi]`.
```python
import numpy as np
from mxnet.contrib.summary import SummaryWriter

x_vals = np.arange(start=0, stop=2 * np.pi, step=0.01)
y_vals = np.sin(x_vals)
with SummaryWriter(logdir='./logs') as sw:
    for x, y in zip(x_vals, y_vals):
        sw.add_scalar(tag='y=sin(x/100)', value=y, global_step=x * 100)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_scalar_sin.png)


### Histogram
We can visulize the value distributions of tensors by logging `NDArray`s in terms of histograms.
The following code snippet generates a series of normal distributions with smaller and smaller standard deviations.
```python
import mxnet as mx
from mxnet.contrib.summary import SummaryWriter


with SummaryWriter(logdir='./logs') as sw:
    for i in range(10):
        data = mx.nd.normal(loc=0, scale=10.0/(i+1), shape=(10, 3, 8, 8))
        sw.add_histogram(tag='norml_dist', values=data, bins=200, global_step=i)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_histogram_norm.png)


### Image
The image logging API can take MXNet `NDArray` or `numpy.ndarray` of 2-4 dimensions.
It will preprocess the input image and write the processed image to the event file.
When the input image data is 2D or 3D, it represents a single image.
When the input image data is a 4D tensor, which represents a batch of images, the logging
API would make a grid of those images by stitching them together before write
them to the event file. The following code snippet saves 15 same images
for visualization in TensorBoard.
```python
import mxnet as mx
import numpy as np
from mxnet.contrib.summary import SummaryWriter
from scipy import misc

face = misc.face().transpose((2, 0, 1))
face = face.reshape((1,) + face.shape)
faces = [face] * 15
faces = np.concatenate(faces, axis=0)

img = mx.nd.array(faces, dtype=faces.dtype)
with SummaryWriter(logdir='./logs') as sw:
    sw.add_image(tag='faces', image=img)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_image_faces.png)


### Embedding
Embedding visualization enables people to get an intuition on how data is clustered
in 2D or 3D space. The following code takes 2,560 images of handwritten digits
from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and log them
as embedding vectors with labels and original images.
```python
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.contrib.summary import SummaryWriter


batch_size = 128


def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32)/255
    return data, label


train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

initialized = False
embedding = None
labels = None
images = None

for i, (data, label) in enumerate(train_data):
    if i >= 20:
        break
    if initialized:
        embedding = mx.nd.concat(*(embedding, data), dim=0)
        labels = mx.nd.concat(*(labels, label), dim=0)
        images = mx.nd.concat(*(images, data.reshape(batch_size, 1, 28, 28)), dim=0)
    else:
        embedding = data
        labels = label
        images = data.reshape(batch_size, 1, 28, 28)
        initialized = True

with SummaryWriter(logdir='./logs') as sw:
    sw.add_embedding(tag='mnist', embedding=embedding, labels=labels, images=images)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_embedding_mnist.png)


### PR Curve
Precision-Recall is a useful metric of success of prediction when the categories are imbalanced.
The relationship between recall and precision can be visualized in terms of precision-recall curves.
The following code snippet logs the data of predictions and labels for visualizing
the precision-recall curve in TensorBoard. It generates 100 numbers uniformly distributed in range `[0, 1]` representing
the predictions of 100 examples. The labels are also generated randomly by picking either 0 or 1.
```python
import mxnet as mx
import numpy as np
from mxnet.contrib.summary import SummaryWriter

with SummaryWriter(logdir='./logs') as sw:
    predictions = mx.nd.uniform(low=0, high=1, shape=(100,), dtype=np.float32)
    labels = mx.nd.uniform(low=0, high=2, shape=(100,), dtype=np.float32).astype(np.int32)
    sw.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
```
![png](https://github.com/reminisce/web-data/blob/tensorboard_doc/mxnet/tensorboard/doc/summary_pr_curve_uniform.png)


## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. autoclass:: mxnet.contrib.summary.SummaryWriter
    :members:
```
<script>auto_index("api-reference");</script>