MXNet Python Data Loading API
=============================
* [Introduction](#introduction) introduces the main feature of data loader in MXNet.
* [Parameters For Data Iterator](#parameters-for-data-iterator) clarifies the different usages for dataiter parameters.
* [Create A Data Iterator](#create-a-data-iterator) introduces how to create a data iterator in MXNet python.
* [How To Get Data](#how-to-get-data) introduces the data resource and data preparation tools.
* [IO API Reference](#io-api-reference) reference for the IO API and their explanation.

Introduction
------------
This page will introduce data input method in MXNet. MXNet use iterator to provide data to the neural network.  Iterators do some preprocessing and generate batch for the neural network.

* We provide basic iterators for MNIST image and RecordIO image.
* To hide the IO cost, prefetch strategy is used to allow parallelism of learning process and data fetching. Data will automatically fetched by an independent thread.

Parameters For Data Iterator
----------------------------

Generally to create a data iterator, you need to provide five kinds of parameters:

* **Dataset Param** gives the basic information for the dataset, e.g. file path, input shape.
* **Batch Param** gives the information to form a batch, e.g. batch size.
* **Augmentation Param** tells which augmentation operations(e.g. crop, mirror) should be taken on an input image.
* **Backend Param** controls the behavior of the backend threads to hide data loading cost.
* **Auxiliary Param** provides options to help checking and debugging.

Usually, **Dataset Param** and **Batch Param** MUST be given, otherwise data batch can't be create. Other parameters can be given according to algorithm and performance need. Examples and detail explanation of the options will be provided in the later Section.

Create A Data Iterator
----------------------
The IO API provides a simple way for you to create data iterator in python.
The following code gives an example of creating a Cifar data iterator.

```python
>>>dataiter = mx.io.ImageRecordIter(
>>>        # Utility Parameter
>>>        # Optional
>>>        # Name of the data, should match the name of the data input of the network
>>>        # data_name='data',
>>>        # Utility Parameter
>>>        # Optional
>>>        # Name of the label, should match the name of the label parameter of the network.
>>>        # Usually, if the loss layer is named 'foo', then the label input has the name
>>>        # 'foo_label', unless overwritten
>>>        # label_name='softmax_label',
>>>        # Dataset Parameter
>>>        # Impulsary
>>>        # indicating the data file, please check the data is already there
>>>        path_imgrec="data/cifar/train.rec",
>>>        # Dataset Parameter
>>>        # Impulsary
>>>        # indicating the image size after preprocessing
>>>        data_shape=(3,28,28),
>>>        # Batch Parameter
>>>        # Impulsary
>>>        # tells how many images in a batch
>>>        batch_size=100,
>>>        # Augmentation Parameter
>>>        # Optional
>>>        # when offers mean_img, each image will substract the mean value at each pixel
>>>        mean_img="data/cifar/cifar10_mean.bin",
>>>        # Augmentation Parameter
>>>        # Optional
>>>        # randomly crop a patch of the data_shape from the original image
>>>        rand_crop=True,
>>>        # Augmentation Parameter
>>>        # Optional
>>>        # randomly mirror the image horizontally
>>>        rand_mirror=True,
>>>        # Augmentation Parameter
>>>        # Optional
>>>        # randomly shuffle the data
>>>        shuffle=False,
>>>        # Backend Parameter
>>>        # Optional
>>>        # Preprocessing thread number
>>>        preprocess_threads=4,
>>>        # Backend Parameter
>>>        # Optional
>>>        # Prefetch buffer size
>>>        prefetch_buffer=1)
```

From the above code, we could find how to create a data iterator. First, you need to explicitly point out what kind of data(MNIST, ImageRecord etc) to be fetched. Then provide the options about the dataset, batching, image augmentation, multi-tread processing and prefetching. Our code will automatically check the validity of the params, if a compulsary param is missing, an error will occur.

How To Get Data
---------------

We provide the [script](../../tests/python/common/get_data.py) to download MNIST data and Cifar10 ImageRecord data. If you would like to create your own dataset, Image RecordIO data format is recommended.

## Create Dataset Using RecordIO

RecordIO implements a file format for a sequence of records. We recommend storing images as records and pack them together. The benefits are:

* Storing images in compacted format, e.g. JPEG, for records can have different size. Compacted format will greatly reduce the dataset size in disk.
* Packing data together allow continous reading on the disk.
* RecordIO has a simple way of partition, which makes it easier for distributed setting. Example about this will be provided later.

We provide the [im2rec tool](../../tools/im2rec.cc) to create Image RecordIO dataset by yourself. Here's the walkthrough:

### 0.Before you start
Make sure you have downloaded the data. You don't need to resize the images by yourself, currently ```im2rec``` could resize it automatically. You could check the promoting message of ```im2rec``` for details.

### 1.Make the image list
After you get the data, you need to make a image list file first.  The format is
```
integer_image_index \t label_index \t path_to_image
```
In general, the program will take a list of names of all image, shuffle them, then separate them into training files name list and testing file name list. Write down the list in the format.

A sample file is provided here
```bash
895099  464     n04467665_17283.JPEG
10025081        412     ILSVRC2010_val_00025082.JPEG
74181   789     n01915811_2739.JPEG
10035553        859     ILSVRC2010_val_00035554.JPEG
10048727        929     ILSVRC2010_val_00048728.JPEG
94028   924     n01980166_4956.JPEG
1080682 650     n11807979_571.JPEG
972457  633     n07723039_1627.JPEG
7534    11      n01630670_4486.JPEG
1191261 249     n12407079_5106.JPEG

```

### 2.Make the binary file
To generate binary image, you need to use *im2rec* in the tool folder. The im2rec will take the path of _image list file_ you generated just now, _root path_ of the images and the _output file path_ as input. These processes usually take several hours, so be patient. :)

A sample command:
```bash
./bin/im2rec image.lst image_root_dir output.bin resize=256
```
More details can be found by running ```./bin/im2rec```.

### Extension: Mutliple Labels for a Single Image

The `im2rec` tool and `mx.io.ImageRecordIter` also has a mutli-label support for a single image.
Assume you have 4 labels for a single image, you can take the following steps to utilize the RecordIO tools.

1. Write the the image list files as follows:
```
integer_image_index \t label_1 \t label_2 \t label_3 \t label_4 \t path_to_image
```

2. When use `im2rec` tools, add a 'label_width=4' to the command argument, e.g.
```bash
./bin/im2rec image.lst image_root_dir output.bin resize=256 label_width=4
```

3. In your iterator generation code, set `label_width=4` and `path_imglist=<<The PATH TO YOUR image.lst>>`, e.g.

```python
dataiter = mx.io.ImageRecordIter(
  path_imgrec="data/cifar/train.rec",
  data_shape=(3,28,28),
  path_imglist="data/cifar/image.lst",
  label_width=4
)
```

Then you're all set for a multi-label image iterator.

```eval_rst
.. raw:: html

    <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>
```


IO API Reference
----------------

```eval_rst
.. automodule:: mxnet.io
    :members:

.. raw:: html

    <script>auto_index("mxnet.io");</script>
```
