# Designing Efficient Data Loaders for Deep Learning

Data loading is an important component of any machine learning system.
When we work with tiny datasets, we can get away with loading an entire dataset into GPU memory.
With larger datasets, we must store examples in main memory.
And when datasets grow too large to fit into main memory,
data loading can become performance-critical.
In designing a data loader,
we aim to achieve more efficient data loading,
to spend less effort on data preparation,
and to present a clean and flexible interface.

We organize this design note as follows:

* **IO Design Insight:**  Guiding principles in data loading design.
* **Data Format:** Our solution using dmlc-core's binary recordIO implementation.
* **Data Loading:** Our method to reduce IO cost by utilizing the threaded iterator provided by dmlc-core.
* **Interface Design:** Our approach to facilitate writing MXNet data iterators in just a few lines of Python.
* **Future Extension:** Prospective ideas for making data loading more flexible.

Our analysis will motivate several requirements that an effective IO system should fulfill.

***List of Key Requirements***
- Small file size.
- Parallel (distributed) packing of data.
- Fast data loading and online augmentation.
- Quick reads from arbitrary parts of the dataset in the distributed setting.

## Design Insight
To design an IO system, we must address two kinds of tasks:
data preparation and data loading.
Data preparation is usually performed offline,
whereas data loading influences the online performance.
In this section, we will introduce our insight of IO design involving the two phases.

### Data Preparation
Data preparation describes the process of packing data
into a desired format for later processing.
When working with large datasets like ImageNet, this process can be time-consuming.
In these cases, there are several heuristics we ought to follow:

- Pack the dataset into small numbers of files. A dataset may contain millions of data instances. Packed data distributes easily from machine to machine.
- Do the packing once. We don't want to repack data every time run-time settings, like the number of machines, are changed.
- Process the packing in parallel to save time.
- Be able to access arbitrary parts of the data easily. This is crucial for distributed machine learning when data parallelism is introduced. Things may get tricky when the data has been packed into several physical data files. The desired behavior could be: the packed data can be logically separated into arbitrary numbers of partitions, no matter how many physical data files there are. For example, if we pack 1000 images into 4 physical files, then each file contains 250 images. If we then use 10 machines to train a DNN, we should be able to load approximately 100 images per machine. Some machines may need images from different physical files.

### Data Loading
The next step to consider is how to load the packed data into RAM.
Our goal is to load the data as quickly as possible.
There are several heuristics we try to follow:
- **Read continuously:** We can read faster when reading from contiguous locations on disk.
- **Reduce the bytes to be loaded:** We can achieve this by storing data in a compact way, e.g. saving images in JPEG format.
- **Load and train in different threads:** This avoids computational bottlenecks while loading data.
- **Save RAM:** Judiciously decide whether to load entire files into RAM.

## Data Format

Since the training of deep neural network often involves large amounts of data,
the format we choose should be both efficient and convenient.
To achieve our goals, we need to pack binary data into a splittable format.
In MXNet, we rely on the binary recordIO format implemented in dmlc-core.

### Binary Record

![baserecordio](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/baserecordio.jpg)
In MXNet's binary RecordIO, we store each data instance as a record.
**kMagic** is a *magic number* indicating the start of a record.
**Lrecord** encodes length and a continue flag.
In lrecord,  
- cflag == 0: this is a complete record
- cflag == 1: start of a multiple-records
- cflag == 2: middle of multiple-records
- cflag == 3: end of multiple-records

**Data** is the space to save data content.
**Pad** is simply a padding space to make record align to 4 bytes.

After we pack the data, each file contains multiple records.
Then, loading can be continuous.
This avoids the low performance that can result
from reading random locations on disk.

One advantage of storing data via records
is that each record can vary in length.
This allows us to save data compactly
when good compression algorithms are available for our data.
For example, we can use JPEG format to save image data.
The packed data will be much smaller
compared with storing uncompressed RGB values for each pixel.

Take ImageNet_1K dataset as an example.
If we store the data as 3 * 256 * 256 array of raw RGB values,
the dataset would occupy more than **200G**.
But after compressing the images using JPEG,
they only occupy about **35G** of disk space.
This significantly reduces the cost owing to reading from disk.

Here's an example of binary recordIO:
![baserecordio](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/ImageRecordIO.jpg)
We first resize the image into 256 * 256,
then compress into JPEG format.
After that, we save a header that indicates the index and label
for that image to be used when constructing the *Data* field for that record.
We then pack several images together into a file.

### Access Arbitrary Parts Of Data

One desirable property for a data loader might be:
The packed data can be logically sliced into an arbitrary number of partitions,
no matter how many physical packed data files there are.
Since binary recordIO can easily locate
the start and end of a record using the Magic Number,
we can achieve the above goal using the InputSplit
functionality provided by dmlc-core.

InputSplit takes the following parameters:
- FileSystem *filesys*: dmlc-core wrapper around the IO operations for different file systems, like hdfs, s3, local. User shouldn't need to worry about the difference between file systems anymore.
- Char *uri*: The URI of files. Note that it could be a list of files because we may pack the data into several physical parts. File URIs are separated by ';'.
- Unsigned *nsplit*: The number of logical splits. *nsplit* could be different from the number of physical files.
- Unsigned *rank*: Which split to load in this process.

The splitting process is demonstrated below:
- Determine the size of each partition.

![beforepartition](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/beforepartition.jpg)

- Approximately partition the records according to file size. Note that the boundary of each part may be located in the middle of a record.

![approxipartition](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/approximatepartition.jpg)

-  Set the beginning of partitions in such a way as to avoid splitting records across partitions.

![afterpartition](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/afterpartition.jpg)

By conducting the above operations,
we now identify the records belong to each part,
and the physical data files needed by each logical part.
InputSplit greatly simplifies data parallelism,
where each process only reads part of the data.

Since our partitioning scheme does not depend on the number of physical data files,
we can process a huge dataset like ImageNet_22K in parallel fashion as illustrated below.
We don't need to consider distributed loading issue at the preparation time,
just select the most efficient physical file number
according to the dataset size and computing resources available.
![parallelprepare](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/parallelprepare.jpg)

## Data Loading and Preprocessing

When the speed of loading and preprocessing can't keep up
with the speed of training or evaluation,
IO can bottleneck the speed of the whole system.
In this section, we will introduce a few tricks
to achieve greater efficiency when loading
and preprocessing data packed in binary recordIO format.
When applied to the ImageNet dataset, our approach achieves
the IO speed of **3000** images/sec **with a normal HDD**.

### Loading and preprocessing on the fly

When training deep neural networks,
we sometimes must load and preprocess the data
while simultaneously training for the following reasons:
- When the whole size of the dataset exceeds available RAM size, we can't load it in advance;
- Sometimes, to make models robust to things like translations, rotations, and small amounts of color shift of noise, we introduce randomness into the training process. In these cases we must re-preprocess the data each time we revisit an example.

In service of efficiency, we also address multi-threading techniques. Taking Imagenet training as an example, after loading a bunch of image records, we can start multiple threads to simultaneously perform image decoding and image augmentation. We depict this process in the following illustration:
![process](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/process.jpg)

### Hide IO Cost Using Threadediter

One way to lower IO cost is to pre-fetch the data for next batch on one thread,
while the main thread performs the forward and backward passes for training.
To support more complicated training schemes,
MXNet provides a more general IO processing pipeline
using *threadediter* provided by dmlc-core.
The key of *threadediter* is to start a stand-alone thread that acts as a data provider,
while the main thread acts as a data consumer as illustrated below.

The threadediter maintains a buffer of a certain size
and automatically fills the buffer when it's not full.
And after the consumer finishes consuming part of the data in the buffer,
threadediter will reuse the space to save the next part of data.
![threadediter](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/threadediter.png)

## MXNet IO Python Interface
We make the IO object as an iterator in numpy.
By achieving that, the user can easily access the data
using a for-loop or calling next() function.
Defining a data iterator is very similar to defining a symbolic operator in MXNet.

The following example code demonstrates a Cifar data iterator.

```python
dataiter = mx.io.ImageRecordIter(
    # Dataset Parameter, indicating the data file, please check the data is already there
    path_imgrec="data/cifar/train.rec",
    # Dataset Parameter, indicating the image size after preprocessing
    data_shape=(3,28,28),
    # Batch Parameter, tells how many images in a batch
    batch_size=100,
    # Augmentation Parameter, when offers mean_img, each image will subtract the mean value at each pixel
    mean_img="data/cifar/cifar10_mean.bin",
    # Augmentation Parameter, randomly crop a patch of the data_shape from the original image
    rand_crop=True,
    # Augmentation Parameter, randomly mirror the image horizontally
    rand_mirror=True,
    # Augmentation Parameter, randomly shuffle the data
    shuffle=False,
    # Backend Parameter, preprocessing thread number
    preprocess_threads=4,
    # Backend Parameter, prefetch buffer size
    prefetch_buffer=1)
```

Generally, to create a data iterator, you need to provide five kinds of parameters:

* **Dataset Param:** Information needed to access the dataset, e.g. file path, input shape.
* **Batch Param:** Specifies how to form a batch, e.g. batch size.
* **Augmentation Param:** Which augmentation operations (e.g. crop, mirror) should be taken on an input image.
* **Backend Param:** Controls the behavior of the backend threads to hide data loading cost.
* **Auxiliary Param:** Provides options to help with debugging.

Usually, **Dataset Param** and **Batch Param** MUST be given,
otherwise the data batch can't be created.
Other parameters can be given as needed.
Ideally, we should separate the MX Data IO into modules,
some of which might be useful to expose to users, for example:

* **Efficient prefetcher:** allows the user to write a data loader that reads their customized binary format that automatically gets multi-threaded prefetcher support.
* **Data transformer:** image random cropping, mirroring, etc. Allows the users to use those tools, or plug in their own customized transformers (maybe they want to add some specific kind of coherent random noise to data, etc.)

## Future Extensions

In the future, there are some extensions to our data IO
that we might consider adding.
Specifically, we might add specialized support
for applications including image segmentation, object localization, and speech recognition.
More detail will be provided when such applications have been running on MXNet.
