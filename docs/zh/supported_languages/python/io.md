MXNet Python Data Loading API
=============================
* [Introduction](#introduction) 介绍 MXNet 数据加载模块的主要特性.
* [Parameters For Data Iterator](#parameters-for-data-iterator) 阐述清楚 dataIter 的参数的不同用法.
* [Create A Data Iterator](#create-a-data-iterator) 介绍如何在创建一个  python 版本的 MXNet 的 Data Iterator.
* [How To Get Data](#how-to-get-data) 介绍数据源以及数据预处理工具.
* [IO API Reference](#io-api-reference) IO API 参考文档以及它们的解释.

Introduction
------------
这页面介绍 MXNet 的数据输入方式. MXNet 使用迭代器 (iterator)的方式向神经网络输入数据. 迭代器做了一些数据预处理, 同时以 batch 的形式向神经网络提供数据.


* 我们为 MNIST 图像和 RecordIO 图像提供了基本的迭代器.
* 为了掩盖 IO 开销, 我们提供了预处理策略, 它可以让机器学习的过程和取数据的过程并行来做. 我们使用一个单独的线程来做取数据的工作.

Parameters For Data Iterator
----------------------------

一般地讲, 如果你要创建一个数据迭代器, 你需要实现下面讲到的五种参数:

* **Dataset Param** 提供数据集的基本信息, 比如说, 文件路径, 输入的数据的 shape. 
* **Batch Param** 提供构建一个 batch 的信息,  比如说 batch size. 
* **Augmentation Param** 指定输入数据的扩充方式 (e.g. crop, mirror).
* **Backend Param** 控制后端线程掩盖数据加载开销的行为.
* **Auxiliary Param** 提供的可选项, 用来帮助检查和 debug..

通常地讲, **Dataset Param** 和 **Batch Param**  *必须* 提 供, 否则 data batch 无法创建. 其他的参数根据算法和性能的需要来设置.  文档的后半部分会提供解释详尽的例子.

Create A Data Iterator
----------------------
这个 IO API 提供在 python 中创建数据迭代器的简单方式. 下面的代码是如何创建一个 Cifar 的数据迭代器的代码.


```python
>>>dataiter = mx.io.ImageRecordIter(
>>>        # Utility Parameter 
>>>        # 可选
>>>        # Name of the data, should match the name of the data input of the network 
>>>        # data_name='data',
>>>        # Utility Parameter
>>>        # 可选
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
>>>        # 可选
>>>        # when offers mean_img, each image will substract the mean value at each pixel
>>>        mean_img="data/cifar/cifar10_mean.bin",
>>>        # Augmentation Parameter
>>>        # 可选
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

从上面的代码中, 我们可以学到如何创建一个数据迭代器. 首先, 你需要明确的指出需要取哪种类型的数据(MNIST, ImageRecord 等等). 然后, 提供描述数据的可选参数, 比如 batching, 数据扩充方式, 多线程处理, 预取数据.  MNNet 框架会检查参数的有效性, 如果一个必须的参数没有提供, 框架会报错.


How To Get Data
---------------


我们提供了 [脚本](../../tests/python/common/get_data.py) 来下载MNIST数据 和Cifar10 ImageRecord 数据.  如果你要创建你自己的数据集, 我们建议您用RecordIO 作为数据格式.

## Create Dataset Using RecordIO

RecordIO 实现了顺序存储 record 的数据格式. 我们建议图像数据按照 record 的格式来存储和打包到一起. 这样做的有以下几点:


* 将图像储存为压缩过的格式, 比如 JPEG, 因为 record 可以大小不同. 压缩过的格式可以极大的减小储存在硬盘上的数据集大小.
* 将若干 record 打包存储, 可以实现硬盘的连续读取, 避免随机读取硬盘.
* RecordIO 容易分块, 这样分布式处理的设置会更加简单. 后面会有例子具体来说明.

我们提供了 [im2rec tool](../../tools/im2rec.cc) 来让用户自己来生成 RecordIO 格式的数据集.  下面是具体流程:

### 0.Before you start
确定你已经下载了需要的数据集. 你不需要自己来做图像的 resize 操作, 现在 `im2rec` 这个工具可以自动来做这种操作. 你可以查看 `im2rec` 提供的的信息来获取更多的内容.

### 1.Make the image list
当你得到了信息之后, 你首先需要生成一个 image list 的文件. 格式如下
```
integer_image_index \t label_index \t path_to_image
```
通常, 这个程序会读取一个包含所有图像文件名的列表文件,  shuffe 这些文件, 然后将 shuffe 后的图像文件名列表分为训练列表文件和测试列表文件. 按照下面给出的例子的格式存储.

简单的例子文件

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

需要用 *im2rec* 这个程序来生成二进制文件.  im2rec 需要你刚刚生成的 _ image list file _ 的路径, 图像的 _root_ 路径 和 _output file_ 路径作为参数. 这个过程需要花费几个小时, 所以需要耐心. :)


简单的例子:
```bash
./bin/im2rec image.lst image_root_dir output.bin resize=256
```
要想获得更多的用法, 直接运行 ```./bin/im2rec```命令, 会在终端打印出详细的用法.

### Extension: Mutliple Labels for a Single Image

`im2rec` 工具以及 `mx.io.ImageRecordIter` 支持对单个图像打多个标签. 假设你需要为单个图像打四个标签, 你可以按照下面的步骤来使用 RecordIO 相关的工具.

1. 按照下面的格式生成 image list 文件:
```
integer_image_index \t label_1 \t label_2 \t label_3 \t label_4 \t path_to_image
```

2. 使用 `im2rec` 时, 需要增加一个 'label_width=4' 作为命令行参数, 比如.
```bash
./bin/im2rec image.lst image_root_dir output.bin resize=256 label_width=4
```

3. 在你的迭代器初始化的时候, 设置 `label_width=4` 和 `path_imglist=<<The PATH TO YOUR image.lst>>` 作为参数.

```python
dataiter = mx.io.ImageRecordIter(
  path_imgrec="data/cifar/train.rec",
  data_shape=(3,28,28),
  path_imglist="data/cifar/image.lst",
  label_width=4
)
```

这样你就完成了一个多标签的数据迭代器.

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
