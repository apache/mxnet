MXNet Python Data Loading API
=============================
* [Introduction](#introduction) introduces the main feature of data loader in MXNet.
* [Create A Data Iterator](#create-a-data-iterator) introduces how to create a data iterator in MXNet python.
* [Parameters For Data Iterator](#parameters-for-data-iterator) clarifies the different usages for dataiter paramters.
* [How To Get Data](#how-to-get-data) introduces the data resource and data preparation tools.
* [Introducing Image RecordIO](*introducing-image-recordio) introduces the benefits brought by using RecordIO as data format.
* [IO API Reference](#io-api-reference) reference for the IO API and their explanation.

Introduction
------------
This page will introduce data input method in MXNet. MXNet use data iterator to provide data to the neural network.  Iterators do some preprocessing and generate batch for the neural network.

* We provide basic iterators for MNIST image and RecordIO image.
* To hide the IO cost, prefetch strategy is used to allow parallelism of learning process and data fetching. Data will automatically fetched by an independent thread.

Create A Data Iterator
----------------------
The IO API provides a simple way for you to create data iterator with various augmentation and preftech options in python.
The following code gives an example of creating a Cifar data iterator.

```python
>>>train_dataiter = mx.io.ImageRecordIter(
>>>        path_imgrec="data/cifar/train.rec",
>>>        mean_img="data/cifar/cifar_mean.bin",
>>>        rand_crop=True,
>>>        rand_mirror=True,
>>>        shuffle=False,
>>>        input_shape=(3,28,28),
>>>        batch_size=batch_size,
>>>        nthread=4,
>>>        prefetch_capacity=6)
```

And an example of creating a MNIST iterator.
```python
>>>train_dataiter = mx.io.MNISTIter(
>>>        image="data/train-images-idx3-ubyte",
>>>        label="data/train-labels-idx1-ubyte",
>>>        input_shape=(784,),
>>>        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
```

From the above code, we could find how to create a data iterator. First, you need to explicitly point out what kind of data(MNIST, ImageRecord etc) to be fetched. Then provide the options about image augmentation, multi-tread processing and prefetching. Our code will automatically check the validity of the options.

Parameters For Data Iterator
----------------------------

Generally to create a data iterator, you need to provide three kinds of parameters:

* **Dataset Param** gives the basic information for the dataset, e.g. file path, input shape.
* **Batch Param** gives the information to form a batch.
* **Augmentation Param** tells which augmentation operations(e.g. crop, mirror) should be taken on an input image.
* **Backend Param** controls the behavior of backend threads to hide data loading cost.
* **Auxiliary Param** provides options to help checking and debugging.

Detail explanation of the options will be provided in the IO API Reference Section.

How To Get Data
---------------

We provide the [script](../../tests/python/common/get_data.py) to download MNIST data and Cifar10 ImageRecord data.

Introducing Image RecordIO
--------------------------

To be added.

IO API Reference
----------------

```eval_rst
.. automodule:: mxnet.io
    :members:
```
