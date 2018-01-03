## Create a Dataset Using RecordIO

RecordIO implements a file format for a sequence of records. We recommend storing images as records and packing them together. The benefits include:

* Storing images in a compact format--e.g., JPEG, for records--greatly reduces the size of the dataset on the disk.
* Packing data together allows continuous reading on the disk.
* RecordIO has a simple way to partition, simplifying distributed setting. We provide an example later.

We provide the [im2rec tool](https://github.com/dmlc/mxnet/blob/master/tools/im2rec.cc) so you can create an Image RecordIO dataset by yourself. The following walkthrough shows you how.

### Prerequisites
Download the data. You don't need to resize the images manually. You can use ```im2rec``` to resize them automatically. For details, see the "Extension: Using Multiple Labels for a Single Image," later in this topic.

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

Sample command:

```bash
./bin/im2rec image.lst image_root_dir output.bin resize=256
```
For more details, run ```./bin/im2rec```.

### Extension: Multiple Labels for a Single Image

The `im2rec` tool and `mx.io.ImageRecordIter` have multi-label support for a single image.
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

```python
dataiter = mx.io.ImageRecordIter(
  path_imgrec="data/cifar/train.rec",
  data_shape=(3,28,28),
  path_imglist="data/cifar/image.lst",
  label_width=4
)
```
