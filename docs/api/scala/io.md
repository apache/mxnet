# MXNet Scala Data Loading API
This topic introduces the data input method for MXNet. MXNet uses an iterator to provide data to the neural network.  Iterators do some preprocessing and generate batches for the neural network.

MXNet provides basic iterators for MNIST and RecordIO images. To hide the cost of I/O, MXNet uses a prefetch strategy that enables parallelism for the learning process and data fetching. Data is automatically fetched by an independent thread.

Topics:

* [Data Iterator Parameters](#parameters-for-data-iterator) clarifies the different usages for dataiter parameters.
* [Create a Data Iterator](#create-a-data-iterator) introduces how to create a data iterator in MXNet for Scala.
* [How to Get Data](#how-to-get-data) introduces the data resource and data preparation tools.
* [IO API Reference](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.IO$) explains the IO API.


## Data Iterator Parameters

To create a data iterator, you typically need to provide five parameters:

* **Dataset Param** provides basic information about the dataset, e.g., file path, input shape.
* **Batch Param** provides information required to form a batch, e.g., batch size.
* **Augmentation Param** tells MXNet which augmentation operations (e.g., crop or mirror) to perform on an input image.
* **Backend Param** controls the behavior of the back-end threads to hide the cost of data loading.
* **Auxiliary Param** provides options for checking and debugging.

You *must* provide the **Dataset Param** and **Batch Param**, otherwise MXNet can't create the data batch. Provide other parameters as required by your algorithm and performance needs. We provide a detailed explanation and examples of the options later.

## Create a Data Iterator

The IO API provides a simple way to create a data iterator in Scala.
The following example code shows how to create a CIFAR data iterator.

```scala
     val dataiter = IO.ImageRecordIter(Map(
            // Utility Parameter
            // Optional
            // Name of the data, should match the name of the data input of the network
            // data_name='data',
            // Utility Parameter
            // Optional
            // Name of the label, should match the name of the label parameter of the network
            // Usually, if the loss layer is named 'foo', then the label input has the name
            // 'foo_label', unless overwritten
            // label_name='softmax_label',
            // Dataset Parameter
            // Impulsary
            // indicating the data file, please check the data is already there
            "path_imgrec" -> "data/cifar/train.rec",
            // Dataset Parameter
            // Impulsary
            // indicating the image size after preprocessing
            "data_shape" -> "(3,28,28)",
            // Batch Parameter
            // Impulsary
            // tells how many images in a batch
            "batch_size" -> "100",
            // Augmentation Parameter
            // Optional
            // when offers mean_img, each image will subtract the mean value at each pixel
            "mean_img" -> "data/cifar/cifar10_mean.bin",
            // Augmentation Parameter
            // Optional
            // randomly crop a patch of the data_shape from the original image
           "rand_crop" -> "True",
            // Augmentation Parameter
            // Optional
            // randomly mirror the image horizontally
            "rand_mirror" -> "True",
            // Augmentation Parameter
            // Optional
            // randomly shuffle the data
            "shuffle" -> "False",
            // Backend Parameter
            // Optional
            // Preprocessing thread number
            "preprocess_threads" -> "4",
            // Backend Parameter
            // Optional
            // Prefetch buffer size
            "prefetch_buffer" = "1"))
```

First, explicitly specify the kind of data (MNIST, ImageRecord, etc.) to fetch. Then, provide the options for the dataset, batching, image augmentation, multi-tread processing,  and prefetching operations. The code automatically validates the parameters. If a required parameter is missing, MXNet returns an error.

## How to Get Data


We provide [scripts](https://github.com/dmlc/mxnet/tree/master/scala-package/core/scripts) to download MNIST data and CIFAR10 ImageRecord data. If you want to create your own dataset, we recommend using the Image RecordIO data format.

## Create a Dataset Using RecordIO

RecordIO implements a file format for a sequence of records. We recommend storing images as records and packing them together. The benefits include:

* Storing images in a compact format--e.g., JPEG, for records--greatly reduces the size of the dataset on the disk.
* Packing data together allows continuous reading on the disk.
* RecordIO has a simple way to partition, simplifying distributed setting. We provide an example later.

We provide the [im2rec tool](https://github.com/dmlc/mxnet/blob/master/tools/im2rec.cc) so you can create an Image RecordIO dataset by yourself. The following walkthrough shows you how.

### Prerequisites
Download the data. You don't need to resize the images manually. You can use `im2rec` to resize them automatically. For details, see "Extension: Using Multiple Labels for a Single Image," later in this topic.

### Step 1. Make an Image List File
After you download the data, you need to make an image list file.  The format is:

```
    integer_image_index \t label_index \t path_to_image
```
Typically, the program takes the list of names of all of the images, shuffles them, then separates them into two lists: a training filename list and a testing filename list. Write the list in the right format.

This is an example file:

```bash
    95099  464     n04467665_17283.JPEG
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

### Step 2. Create the Binary File
To generate a binary image, use `im2rec` in the tool folder. `im2rec` takes the path of the `_image list file_` you generated, the `_root path_` of the images, and the `_output file path_` as input. This process usually takes several hours, so be patient.

A sample command:

```bash
    ./bin/im2rec image.lst image_root_dir output.bin resize=256
```
For more details, run ```./bin/im2rec```.

### Extension: Multiple Labels for a Single Image

The `im2rec` tool and `IO.ImageRecordIter` have multi-label support for a single image.
For example, if you have four labels for a single image, you can use the following procedure to use the RecordIO tools.

1. Write the image list files as follows:

     ```
         integer_image_index \t label_1 \t label_2 \t   label_3 \t label_4 \t path_to_image
     ```

2. Run `im2rec`, adding a 'label_width=4' to the command argument, for example:

     ```bash
         ./bin/im2rec image.lst image_root_dir output.bin resize=256 label_width=4
     ```

3. In the iterator generation code, set `label_width=4` and `path_imglist=<<The PATH TO YOUR image.lst>>`, for example:

     ```scala
         val dataiter = IO.ImageRecordIter(Map(
           "path_imgrec" -> "data/cifar/train.rec",
           "data_shape" -> "(3,28,28)",
           "path_imglist" -> "data/cifar/image.lst",
           "label_width" -> "4"
         ))
     ```

## Next Steps
* [NDArray API](ndarray.md) for vector/matrix/tensor operations
* [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training
