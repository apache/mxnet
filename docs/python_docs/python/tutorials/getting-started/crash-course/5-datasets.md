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


# Gluon `Dataset`s and `DataLoader`

One of the most critical steps for model training and inference is loading the data: without data you can't learn. In this tutorial, you will use the Gluon API to define a [Dataset](/api/python/docs/api/gluon/data/index.html#datasets) and use a [DataLoader](/api/python/docs/api/gluon/data/index.html#dataloader) to iterate through the dataset in mini-batches.


```python
import mxnet as mx
import os
import time
import tarfile

```

## Introduction to `Dataset`s

[Dataset](/api/python/docs/api/gluon/data/index.html#datasets) objects are used to represent collections of data, and include methods for loading and parsing the data (often stored on disk). Gluon has a number of different `Dataset` classes for working with image data out-of-the-box, but you'll use the [ArrayDataset](/api/python/docs/api/gluon/data/index.html#mxnet.gluon.data.ArrayDataset) to introduce the idea of a `Dataset`.

To start, you can generate random data `X`, consisting of 10 samples and 3 features with the corresponding random labels `y` simulating a typical supervised learning task. Then you can pass these labels to the `ArrayDataset` method for them become a `Dataset` class.





```python
mx.random.seed(42) # Fix the seed for reproducibility
X = mx.random.uniform(shape=(10, 3))
y = mx.random.uniform(shape=(10, 1))
dataset = mx.gluon.data.dataset.ArrayDataset(X, y)
```

A key feature of the `Dataset` class is the __*ability to retrieve a single sample from a given index*__. In this case, your random data and labels were generated in memory, so this `ArrayDataset` doesn't have to load anything from disk, but the interface is the same for any `Dataset` class.




```python

sample_idx = 4
sample = dataset[sample_idx]

assert len(sample) == 2
assert sample[0].shape == (3, )
assert sample[1].shape == (1, )
print(sample)
```

    (
    [0.74707687 0.37641123 0.46362457]
    <NDArray 3 @cpu(0)>, 
    [0.35440788]
    <NDArray 1 @cpu(0)>)



You get a tuple output consisting of a data sample and its corresponding label, which makes sense because you passed the data `X` and the labels `y` in that order when you instantiated the `ArrayDataset`. You don't usually retrieve individual samples from `Dataset` objects though (unless we're quality checking the output samples). Instead you typically use a `DataLoader`.

## Introduction to `DataLoader`

A [DataLoader](/api/python/docs/api/gluon/data/index.html#dataloader) is used to create mini-batches of samples from a [Dataset](/api/python/docs/api/gluon/data/index.html#datasets), and the `DataLoader` provides a convenient iterator interface for looping through these batches. It's typically much more efficient to pass a mini-batch of data through a neural network than a single sample at a time, because the computation can be performed in parallel. A required parameter of `DataLoader` is the size of the mini-batches you want to create, called `batch_size`.

Another benefit of using `DataLoader` is the ability to easily load data in parallel using [multiprocessing](https://docs.python.org/3.6/library/multiprocessing.html). You can set the `num_workers` parameter to the number of CPUs available on your machine for potentially maximal performance, or limit it to a lower number to spare resources. Please note, that sometimes too many CPUs will not necessarily improve your overall efficiency, so feel free to modify `num_workers` to the most efficient value for your dataset.




```python

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()

data_loader = mx.gluon.data.DataLoader(dataset, batch_size=5, num_workers=CPU_COUNT)

for X_batch, y_batch in data_loader:
    print("X_batch has shape {}, and y_batch has shape {}".format(X_batch.shape, y_batch.shape))
```

    X_batch has shape (5, 3), and y_batch has shape (5, 1)
    X_batch has shape (5, 3), and y_batch has shape (5, 1)


You can see 2 mini-batches of data (and labels), each with 5 samples, which makes sense given you started with a dataset of 10 samples. When comparing the shape of the batches to the samples returned by the `Dataset`, you've gained an extra dimension at the start which is typically called the batch axis.

Our `data_loader` loop will stop when every sample of `dataset` has been returned as part of a batch. Sometimes the dataset length isn't divisible by the mini-batch size, leaving a final batch with a smaller number of samples. `DataLoader`'s default behavior is to return this smaller mini-batch, but this can be changed by setting the `last_batch` parameter to `discard` (which ignores the last batch) or `rollover` (which starts the next epoch with the remaining samples).

## Machine learning with `Dataset`s and `DataLoader`s

You will often use a few different `Dataset` objects in your Machine Learning project. It's essential to separate your training dataset from testing dataset, and it's also good practice to have validation dataset (a.k.a. development dataset) that can be used for optimising hyperparameters.

Using Gluon `Dataset` objects, you define the data to be included in each of these separate datasets. Common use cases for loading data are covered already (e.g. [mxnet.gluon.data.vision.datasets.ImageFolderDataset](/api/python/docs/api/gluon/data/vision/index.html)), but it's simple to create your own custom `Dataset` classes for other types of data. You can even use `Dataset` objects for common datasets if you want to experiment quickly; the `Dataset` class can even download and parse the data for you! In this example you use the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset from Zalando Research.

Many of the image `Dataset`'s accept a function (via the optional `transform` parameter) which is applied to each sample returned by the `Dataset`. It's useful for performing data augmentation, but can also be used for more simple data type conversion and pixel value scaling as seen below.




```python

def transform(data, label):
    data = data.astype('float32')/255
    return data, label

train_dataset = mx.gluon.data.vision.datasets.FashionMNIST(train=True)#, transform=transform)
valid_dataset = mx.gluon.data.vision.datasets.FashionMNIST(train=False)#, transform=transform)
```


```python
%matplotlib inline
from matplotlib.pylab import imshow

sample_idx = 234
sample = train_dataset[sample_idx]
data = sample[0]
label = sample[1]
label_desc = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

imshow(data[:,:,0].asnumpy(), cmap='gray')
print("Data type: {}".format(data.dtype))
print("Label: {}".format(label))
print("Label description: {}".format(label_desc[label]))
```

    Data type: <class 'numpy.uint8'>
    Label: 8
    Label description: Bag



![png](output_10_1.png)



```python
# ![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/gluon/datasets/fashion_mnist_bag.png)
```

When training some models it can be important to shuffle the training samples every time you pass through the dataset (i.e. each epoch). Sometimes the order of your samples will have a spurious relationship with the target variable, and shuffling the samples helps remove this, improving performance. With [DataLoader](/api/python/docs/api/gluon/data/index.html#dataloader) it's as simple as adding `shuffle=True`.

If you have more complex shuffling requirements (e.g. when handling sequential data), take a look at [mxnet.gluon.data.BatchSampler](/api/python/docs/api/gluon/data/index.html#mxnet.gluon.data.BatchSampler) and pass this to your `DataLoader` instead.


```python

batch_size = 32
train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=CPU_COUNT)
valid_data_loader = mx.gluon.data.DataLoader(valid_dataset, batch_size, num_workers=CPU_COUNT)
```

With both `DataLoader`s defined, you can now train a model to classify each image and evaluate the validation loss at each epoch. Our Fashion MNIST dataset has 10 classes including shirt, dress, sneakers, etc. In this example, you will define a simple fully connected network with a softmax output and use cross entropy as our loss.


```python
from mxnet import gluon, autograd, ndarray

def construct_net():
    net = gluon.nn.HybridSequential()
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(64, activation="relu"))
    net.add(gluon.nn.Dense(10))
    return net

# construct and initialize network.
ctx =  mx.gpu() if mx.context.num_gpus() else mx.cpu()

net = construct_net()
net.hybridize()
net.initialize(mx.init.Xavier(), ctx=ctx)
# define loss and trainer.
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```


```python

            output = net(data.astype('float32'))
    
            loss = criterion(output, label)
```


```python
epochs = 5
for epoch in range(epochs):
    # training loop (with autograd and trainer steps, etc.)
    cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
    training_samples = 0
    for batch_idx, (data, label) in enumerate(train_data_loader):
        data = data.as_in_context(ctx).reshape((-1, 784)) # 28*28=784
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.astype('float32'))
            loss = criterion(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_train_loss += loss.sum()
        training_samples += data.shape[0]
    train_loss = cumulative_train_loss.asscalar()/training_samples

    # validation loop
    cumulative_valid_loss = mx.nd.zeros(1, ctx)
    valid_samples = 0
    for batch_idx, (data, label) in enumerate(valid_data_loader):
        data = data.as_in_context(ctx).reshape((-1, 784)) # 28*28=784
        label = label.as_in_context(ctx)
        output = net(data.astype('float32'))
        loss = criterion(output, label)
        cumulative_valid_loss += loss.sum()
        valid_samples += data.shape[0]
    valid_loss = cumulative_valid_loss.asscalar()/valid_samples

    print("Epoch {}, training loss: {:.2f}, validation loss: {:.2f}".format(epoch, train_loss, valid_loss))

```

    Epoch 0, training loss: 2.30, validation loss: 2.42
    Epoch 1, training loss: 2.30, validation loss: 2.42
    Epoch 2, training loss: 2.30, validation loss: 2.42
    Epoch 3, training loss: 2.30, validation loss: 2.42
    Epoch 4, training loss: 2.30, validation loss: 2.42


# Using custom data with included `Dataset`s

Gluon has a number of different [Dataset](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataset#mxnet.gluon.data.Dataset) classes for working with your own image data straight out-of-the-box. You can get started quickly using the [mxnet.gluon.data.vision.datasets.ImageFolderDataset](/api/python/docs/api/gluon/data/vision/datasets/index.html#mxnet.gluon.data.vision.datasets.ImageFolderDataset) which loads images directly from a user-defined folder, and infers the label (i.e. class) from the folders.

You will run through an example for image classification, but a similar process applies for other vision tasks. If you already have your own collection of images to work with you could partition your data into training and test sets, and place all objects of the same class into seperate folders. Similar to:
```
    ./images/train/car/abc.jpg
    ./images/train/car/efg.jpg
    ./images/train/bus/hij.jpg
    ./images/train/bus/klm.jpg
    ./images/test/car/xyz.jpg
    ./images/test/bus/uvw.jpg
```

You can download the Caltech 101 dataset if you don't already have images to work with for this example, but please note the download is 126MB.


```python
data_folder = "data"
dataset_name = "101_ObjectCategories"
archive_file = "{}.tar.gz".format(dataset_name)
archive_path = os.path.join(data_folder, archive_file)
data_url = "https://s3.us-east-2.amazonaws.com/mxnet-public/"

if not os.path.isfile(archive_path):
    mx.test_utils.download("{}{}".format(data_url, archive_file), dirname = data_folder)
    print('Extracting {} in {}...'.format(archive_file, data_folder))
    tar = tarfile.open(archive_path, "r:gz")
    tar.extractall(data_folder)
    tar.close()
    print('Data extracted.')
```

    Extracting 101_ObjectCategories.tar.gz in data...
    Data extracted.


After downloading and extracting the data archive, you have two folders: `data/101_ObjectCategories` and `data/101_ObjectCategories_test`. You can load the data into separate training and testing [ImageFolderDataset](/api/python/docs/api/gluon/data/vision/datasets/index.html#mxnet.gluon.data.vision.datasets.ImageFolderDataset)s.

training_path = os.path.join(data_folder, dataset_name)
testing_path = os.path.join(data_folder, "{}_test".format(dataset_name))

You instantiate the [ImageFolderDataset](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=imagefolderdataset#mxnet.gluon.data.vision.datasets.ImageFolderDataset) by providing the path to the data, and the folder structure will be traversed to determine which image classes are available and which images correspond to each class. Make sure the classes used in both the training and in the testing datasets are the same, otherwise the label encodings can get muddled.

Additionally, you can pass a `transform` parameter to these `Dataset`'s as you've previously seen.


```python
cd data
```

    /home/ec2-user/SageMaker/data



```python
!ls
```

    101_ObjectCategories  101_ObjectCategories.tar.gz  101_ObjectCategories_test



```python
training_path='/home/ec2-user/SageMaker/data/101_ObjectCategories'
testing_path='/home/ec2-user/SageMaker/data/101_ObjectCategories_test'
train_dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(training_path)
test_dataset = mx.gluon.data.vision.datasets.ImageFolderDataset(testing_path)
```

Samples from these datasets contain tuples of data with the corresponding labels. Images are loaded from disk, decoded and optionally transformed when the `__getitem__(i)` method is called (equivalent to `train_dataset[i]`).

As with the Fashion MNIST dataset the labels will be integer encoded. You can use the `synsets` property of the [ImageFolderDataset](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=imagefolderdataset#mxnet.gluon.data.vision.datasets.ImageFolderDataset)s to retrieve the original descriptions (e.g. `train_dataset.synsets[i]`).


```python

sample_idx = 539
sample = train_dataset[sample_idx]
data = sample[0]
label = sample[1]

imshow(data.asnumpy(), cmap='gray')
print("Data type: {}".format(data.dtype))
print("Label: {}".format(label))
print("Label description: {}".format(train_dataset.synsets[label]))
assert label == 1
```

    Data type: <class 'numpy.uint8'>
    Label: 1
    Label description: Faces_easy



![png](output_27_1.png)



```python
# ![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/gluon/datasets/caltech101_face.png)<!--notebook-skip-line-->
```

# Using your own data with custom `Dataset`s

Sometimes you have data that doesn't quite fit the format expected by the included [Dataset](/api/python/docs/api/gluon/data/index.html#mxnet.gluon.data.Dataset) classes. You might be able to preprocess your data to fit the expected format, but it may be easier to create your own dataset to do this.

To create your own custom `Dataset` you need to create a class that implements a `__getitem__` method and returns a sample (i.e. a tuple of [mx.nd.NDArray](/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.NDArray)'s).

# New in MXNet 2.0: faster C++ backend DataLoaders

As part of an effort to speed up the current data loading pipeline using gluon dataset and dataloader, a new dataloader was created that uses the C++ backend and avoids potentially slow calls to Python functions.

See [original issue](https://github.com/apache/incubator-mxnet/issues/17269), [pull request](https://github.com/apache/incubator-mxnet/pull/17464) and [implementation](https://github.com/apache/incubator-mxnet/pull/17841).

The data loading pipeline is the major bottleneck for many training tasks. This flow can be summarized as:


```python
| Dataset.__getitem__ -> 
| Transform.__call__()/forward() ->
| Batchify ->
| (optional communicate through shared_mem) ->
| split_and_load(ctxs) ->
| <training on GPUs> -> 
```

Typically, the performance bottlenecks are from slow python dataset retrievals, transform functions, multithreading issues due to the global interpreter lock, Python multiprocessing issues due to speed, and batchify issues due to poor memory management. By using the C++ backend you may be able to remove these bottlenecks and achieve a significant performance gain.

This new dataloader provides: 
- common C++ batchify functions that are split and context aware
- a C++ MultithreadingDataLoader which inherit the same arguments as gluon.data.DataLoader but use MXNet's internal multithreading rather than Python's multiprocessing.
- fallback to python multiprocessing whenever the dataset is not fully supported by the backend in the case that:
    - the transform is not fully hybridizable
    - batchify is not fully supported by backend

Users can continue to use the traditional gluon.data.Dataloader, and the C++ backend will be applied automatically. The 'try_nopython' default is 'Auto', which detects whether the C++ backend is available for the given dataset and transforms. 

Here you can view a performance increase for the CIFAR10 dataset with the C++ backend.

### Using the C++ backend:


```python
cpp_dl = mx.gluon.data.DataLoader(
    mx.gluon.data.vision.CIFAR10(train=True, transform=None), batch_size=32, num_workers=2,try_nopython=True)
```


```python
start = time.time()
for _ in range(3):
    print(len(cpp_dl))
    for _ in cpp_dl:
        pass
print('Elapsed time for backend dataloader:', time.time() - start)
```

    1563
    1563
    1563
    Elapsed time for backend dataloader: 2.421664237976074


### Using the Python backend:


```python
dl = mx.gluon.data.DataLoader(
    mx.gluon.data.vision.CIFAR10(train=True, transform=None), batch_size=32, num_workers=2,try_nopython=False)
```


```python
start = time.time()
for _ in range(3):
    print(len(dl))
    for _ in dl:
        pass
print('Elapsed time for python dataloader:', time.time() - start)
```

    1563
    1563
    1563
    Elapsed time for python dataloader: 6.896752119064331


### The C++ backend loader was almost 3X faster for this particular use case
This improvement in performance will not be seen in all cases, but when possible it's highly encouraged that you compare the dataloader throughput for these two options.


```python
# <!-- INSERT SOURCE DOWNLOAD BUTTONS -->
```
