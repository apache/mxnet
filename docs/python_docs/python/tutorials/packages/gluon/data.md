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

# Data

One of the most critical steps for model training and inference is loading the data: without data you can’t do deep learning! In this tutorial we use the `data` module to:

1) Define a [`Dataset`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html)
2) Use `transform`s to augment the dataset
3) Use a [`DataLoader`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.DataLoader.html) to iterate through the dataset in mini-batches

## Getting started with [`Dataset`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html)s

Our very first step is to define the [`Dataset`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html). A [`Dataset`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html) is MXNet Gluon's interface to the data that's stored on disk (or elsewhere). Out of the box, MXNet Gluon comes with a number of common benchmarking datasets such as CIFAR-10 and MNIST. We'll use an MNIST variant, called FashionMNIST, to understand the role of the [`Dataset`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html). When we instantiate this benchmarking dataset, the data will be downloaded to disk and be ready to use.

```
import mxnet as mx

dataset = mx.gluon.data.vision.FashionMNIST()
```

So how do we get data out of a [`Dataset`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html)? We index.

```
sample = dataset[42]
```

Choosing a random index of 42 above, we get back a single sample from the dataset. A single sample usually contains multiple elements.

```
len(sample)
```

Our sample has 2 elements in this case. We have the image as the first element and the label as the second. Although this is a common pattern, you should check the documentation (or implementation) of the dataset you're using.

We can unpack the sample into `data` and `label` and visualise our single sample.

```
data_sample, label_sample = sample
print('Clothing Category #{}'.format(label_sample))
```

```
import matplotlib.pyplot as plt

plot = lambda s: plt.imshow(s[:,:,0].asnumpy(), cmap='gray')
plot(data_sample)
```

We can access all of our samples in the dataset using the indexing method above, but usually we leave this to the [`DataLoader`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.DataLoader.html). We'll see an example of this shortly, but first we'll see how to change our dataset with `transform`s.

## Getting started with `transform`s

Similar to `dataset`s which return individual samples, `transform`s give us a way to change individual samples of our `dataset`.

#### Why would you want to do this?

Most of the time, samples returned by the dataset aren't in quite the right format to be passed into a neural network. One common example is to have samples with a data type of `int8` when your network expects `float32`. Another common example with images is when you have an image with a dimension order of (height, width, channel) when you need (channel, height, width) for the network.

Augmenting samples is also common practice: that is, randomly permuting existing samples to make new samples to reduce issues of network overfitting. Using image samples as an example, you could crop to random regions, flip from left to right or even jitter the brightness of the image.

#### And how do you do it?

We first define our transform and then apply it to our dataset.

MXNet Gluon has a number of in-build transforms available at `mxnet.gluon.data.vision.transforms`, so let's use [`RandomResizedCrop`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.vision.transforms.RandomResizedCrop.html) as an example. We're specifying that we want a random crop that contains 60% to 80% of the original image, and then we want to scale this cropped region to 20px by 20px. You should notice we're instantiating a class that can be called like a function.

```
from mxnet.gluon.data.vision.transforms import RandomResizedCrop

transform_fn = RandomResizedCrop(size=(20, 20), scale=(0.6, 0.8))
```

After defining `transform_fn`, we now need to apply it to the dataset. We call the `transform_first` method of our dataset to apply `transform_fn` to the first element of all samples. We had 2 elements per sample in our example, so we only apply `transform_fn` to the image and not the label.

Advanced: `transform`, instead of `transform_first` can be used to transform all elements.

```
dataset = dataset.transform_first(transform_fn)
```

When we retrieve the same sample as before from the dataset, we now see an augmented version of the image (with the same label). We'd see a different augmented image every time we retrieve this sample because the transform is applied lazily by default.

```
data_sample, label_sample = dataset[42]
print('Clothing Category #{}'.format(label_sample))
plot(data_sample)
```

## Getting started with [`DataLoader`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.DataLoader.html)s

Our `dataset` gives us individual samples, but we usually train neural networks on batches of samples. A [`DataLoader`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.DataLoader.html) retrieves samples from the `dataset` and stacks them into batches. At a minimum, all we need to specify is the number of samples we want in each batch, called the `batch_size`, but [`DataLoader`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.DataLoader.html)s have many other useful features. Shuffling data samples for training is as simple as setting `shuffle`. 

Advanced: A larger `batch_size` often speeds up training, but can lead to out-of-memory and convergence issues. 12345 is just for demonstration purposes: you'll more commonly use a smaller batch size (e.g. 128) but it depends on the model.

```
dataloader = mx.gluon.data.DataLoader(dataset,
                                      batch_size=12345,
                                      shuffle=True)
```

We iterate through the `dataloader` to get all the batches in our `dataset`, and we usually place the code for network training inside this loop. You'll notice that `data_batch` has an extra dimension at the start (compared with `data_sample`): this is often called the batch dimension.

```
for batch_idx, (data_batch, label_batch) in enumerate(dataloader):
    print('Batch {} has shape {}'.format(batch_idx, data_batch.shape))
```

You might have noticed that the last batch contains fewer samples than the others. We had a remainder at the end and by default a [`DataLoader`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.DataLoader.html) keeps the incomplete batch (see `last_batch` argument).

We can plot a few (augmented) images from the last batch.

```
plt.figure(1)
plt.subplot(131); plot(data_batch[123])
plt.subplot(132); plot(data_batch[234])
plt.subplot(133); plot(data_batch[345])
plt.show()
```

## Conclusion

We've now seen all of the core components of the MXNet Gluon data pipeline. You should now understand the difference between [`Dataset`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.Dataset.html)s, that return individual samples, and [`DataLoader`](/api/python/docs/api/gluon/_autogen/mxnet.gluon.data.DataLoader.html)s, that return batches of samples. You should also be able to transform your `dataset` for pre-processing or augmentation purposes.

## Recommended Next Steps

We used presets and a lot of default values in this tutorial, but there are many other options and possibilities. You're likely to want to use your own dataset for training. You can learn more about this in the dedicated `Dataset` tutorial. We also have a tutorial on `transform`s with examples on how to compose transform operations. And last but not least, we have a very useful tutorial on `DataLoader`s. You'll often find that loading data can be the bottleneck to training, and this tutorial contains some useful tricks to speed up the data pipeline.
