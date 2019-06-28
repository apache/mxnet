<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Data Transforms

Creating a [`Dataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataset#mxnet.gluon.data.Dataset) is the starting point of the data pipeline, but we usually need to change samples before passing them to the network. Gluon `transforms` provide us with a simple way to apply these changes. We can use out-of-the-box transforms or create our own.

We'll demonstrate this by adjusting samples returned by the [`CIFAR10`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=cifar#mxnet.gluon.data.vision.datasets.CIFAR10) dataset and start by importing the relevant modules.


```python
import mxnet as mx
from matplotlib import pyplot as plt
from mxnet import image
from mxnet.gluon import data as gdata, utils
import numpy as np
```

After creating our [CIFAR-10 `Dataset`]([`CIFAR10`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=cifar#mxnet.gluon.data.vision.datasets.CIFAR10)), we can inspect a random sample.


```python
dataset = mx.gluon.data.vision.CIFAR10()
```


```python
sample_idx = 42
sample_data, sample_label = dataset[sample_idx]
print("data shape: {}".format(sample_data.shape))
print("data type: {}".format(sample_data.dtype))
print("data range: {} to {}".format(sample_data.min().asscalar(),
                                    sample_data.max().asscalar()))
print("label: {}".format(sample_label))
plt.imshow(sample_data.asnumpy())
```

Our sample looks fine, but we need to need to make a few changes before using this as an input to a neural network.

### Using `ToTensor` and `.transform_first`

Ordering of dimensions (sometimes called the data layout) is important for correct usage of a neural network. Currently our samples are ordered (height, width, channel) but we need to change this to (channel, height, width) before passing to our network. We also need to change our data type. Currently it's `uint8`, but we need to change this to `float32`.

MXNet Gluon provides a number of useful transforms for common computer vision cases like this. We will use [`ToTensor`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=totens#mxnet.gluon.data.vision.transforms.ToTensor) to change the data layout and convert integers (between 0 and 255) to floats (between 0 and 1). We apply the transform to our `dataset` using the [`transform_first`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=transform_first#mxnet.gluon.data.Dataset.transform_first) method. We have 2 elements per sample here (i.e. data and label), so the transform is only applied to the first element (i.e. data).

Advanced: `transform` (instead of [`transform_first`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=transform_first#mxnet.gluon.data.Dataset.transform_first)) can be used to transform all elements in the sample.


```python
transform_fn = mx.gluon.data.vision.transforms.ToTensor()
dataset = dataset.transform_first(transform_fn)
```


```python
sample_data, sample_label = dataset[sample_idx]
print("data shape: {}".format(sample_data.shape))
print("data type: {}".format(sample_data.dtype))
print("data range: {} to {}".format(sample_data.min().asscalar(),
                                    sample_data.max().asscalar()))
print("label: {}".format(sample_label))
```

Our data has changed, while the label has been left untouched.

### `Normalize`

We scaled the values of our data samples between 0 and 1 as part of [`ToTensor`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=totens#mxnet.gluon.data.vision.transforms.ToTensor) but we may want or need to normalize our data instead: i.e. shift to zero-center and scale to unit variance. You can do this with the following steps:

* **Step 1** is to calculate the mean and standard deviation of each channel for the entire training dataset.
* **Step 2** is to use these statistics to normalize each sample for training and for inference too.

When using pre-trained models, you need to use the same normalization statistics that were used for training. Models from the Gluon Model Zoo expect normalization statistics of `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.

When training your own model from scratch, you can estimate the normalization statistics from the training dataset (not test dataset) using the code snippet found below. Models often benefit from input normalization because it prevents saturation of activations and prevents certain features from dominating due to differences in scale.


```python
# estimate channel mean and std
sample_means = np.empty(shape=(len(dataset), 3))
sample_stds = np.empty(shape=(len(dataset), 3))
for idx in range(len(dataset)):
    sample_data = dataset[idx][0].asnumpy()
    sample_means[idx] = sample_data.mean(axis=(1,2))
    sample_stds[idx] = sample_data.std(axis=(1,2))
print("channel means: {}".format(sample_means.mean(axis=0)))
print("channel stds: {}".format(sample_stds.mean(axis=0)))
```

We can create our `Normalize` transform using these statistics and apply it to our `dataset` as before.


```python
normalize_fn = mx.gluon.data.vision.transforms.Normalize(mean=[0.49139969, 0.48215842, 0.44653093],
                                                         std=[0.20220212, 0.19931542, 0.20086347])
dataset = dataset.transform_first(normalize_fn)
```

And now when we inspect the values for a single sample, we find that it's no longer bounded by 0 and 1.


```python
sample_data, sample_label = dataset[sample_idx]
print("data range: {} to {}".format(sample_data.min().asscalar(),
                                    sample_data.max().asscalar()))
```

### `Compose`

We've now seen two examples of transforms: [`ToTensor`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=totens#mxnet.gluon.data.vision.transforms.ToTensor) and `Normalize`. We applied both transforms to our `dataset` through repeated calls to `transform_first`, but Gluon has a dedicated transform to stack other transforms that's preferred. With [`Compose`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=compose#mxnet.gluon.data.vision.transforms.Compose) we can choose and order the transforms we want to apply.

Caution: ordering of transforms is important. e.g. [`ToTensor`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=totens#mxnet.gluon.data.vision.transforms.ToTensor) should be applied before `Normalize`, but after `Resize` and `CenterCrop`.


```python
dataset = mx.gluon.data.vision.CIFAR10()
transform_fn = mx.gluon.data.vision.transforms.Compose([
    mx.gluon.data.vision.transforms.ToTensor(),
    mx.gluon.data.vision.transforms.Normalize(mean=[0.49139969, 0.48215842, 0.44653093],
                                              std=[0.20220212, 0.19931542, 0.20086347]),
])
dataset = dataset.transform_first(transform_fn)
```


```python
sample_data, sample_label = dataset[sample_idx]
print("data range: {} to {}".format(sample_data.min().asscalar(),
                                    sample_data.max().asscalar()))
```

As a sanity check, we get the same result as before.

### `CenterCrop` and `Resize`

Some networks require an input of a certain size (e.g. convolutional neural network with a final dense layer). Specifying the `size`, you can use `CenterCrop` and `Resize` to modify the spatial dimensions of an image. We show an example of downsampling to 10px by 10px here.


```python
dataset = mx.gluon.data.vision.CIFAR10()
transform_fn = mx.gluon.data.vision.transforms.Resize(size=(10, 10))
dataset = dataset.transform_first(transform_fn)
sample_data, sample_label = dataset[sample_idx]
plt.imshow(sample_data.asnumpy())
```

### Augmentation

Augmenting samples is also common practice: that is, randomly permuting existing samples to make new samples to reduce issues of network overfitting. Using image samples as an example, you could crop to random regions, flip from left to right or even jitter the brightness of the image. Gluon has a number of random transforms that are covered in depth in the Data Augmentation tutorial.

## Summary

We've now seen how to use Gluon transforms to adjust samples returned by a [`Dataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataset#mxnet.gluon.data.Dataset). You should aim to construct a `transform_fn` using [`Compose`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=compose#mxnet.gluon.data.vision.transforms.Compose), and then apply it to the `Dataset` using [`transform_first`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=transform_first#mxnet.gluon.data.Dataset.transform_first) or `transform`.

### Additional Reading

Check out the introduction to data tutorial for an overview of how `transforms` fit into the complete data pipeline. More information on data augmentation can be found here. And the [GluonNLP](https://gluon-nlp.mxnet.io/api/modules/data.html) and [GluonCV](https://gluon-cv.mxnet.io/api/data.transforms.html) toolkits provide a variety of domain specific transforms that you might find useful.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
