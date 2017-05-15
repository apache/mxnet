
In this tutorial we focus on how to feed data into a training and inference
program. Most training and inference modules in MXNet accepts data iterators,
which simplifies this procedure, especially when reading large datasets from
filesystems. Here we discuss the API conventions and several provided iterators.

## MXNet Data Iterator  
Data Iterators in *MXNet* are similar to the built-in function `iter` in ``Python``. In ``Python`` `iter` allows to fetch items sequentially by calling  `next()` on __iterable__ collection objects such as a Python `list`. `iter` provides a abstract interface for traverising various types of __iterable__ collections without needing to expose the underlying data structure.  

In MXNet, __Iterators__ return a batch of data `DataBatch` on each call to `next`.
A `DataBatch` often contains *n* training examples and their corresponding labels. Here *n* is the `batch_size` of the Iterator. At the end of the data stream when there is no more data to read, the Iterator raises ``StopIteration`` exception like *Python* `iter`.  
The structure of `DataBatch` is defined in [DataBatch](http://mxnet.io/api/python/io.html#mxnet.io.DataBatch).
     
All IO in *MXNet* is handled via mx.io.DataIter and its subclasses. We will see below a few commonly used Iterators provided by *MXNet*.

Setup environment:

A data iterator returns a batch of data in each `next` call.
A batch often contains *n* examples and the according labels. Here *n* is
called as the batch size.

The following codes define a simple data batch that is able to be read by most
training/inference modules.

```python
class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
```

We explain what each attribute means:

- `data` is a list of `NDArray`, each array contains *n* examples. For
  instance, if an example is presented by a length `k` vector, then the shape of
  the array will be `(n, k)`.

  Each array will be copied into a free variable such as created by
  `mx.sym.Variable()` later. The mapping from arrays to free variables should be
  given by the `provide_data` attribute of the iterator, which will be discussed
  shortly.

- `label` is also a list of `NDArray`. Often each array is a 1-dimensional
  array with shape `(n,)`. For classification, each class is represented by an
  integer starting from 0.

- `pad` is an integer which shows the number of examples added in the last of the
  batch that are merely used for padding. These examples should be ignored in
  the results, such as computing the gradient. A nonzero padding is often used
  when we reach the end of the data and the total number of examples cannot be
  divided by the batch size.

### Data Variables

Before showing the data iterator, we first discuss how to find free variables in
a symbol. A symbol often contains one or more explicit free variables and also
implicit ones.

The following code defines a multilayer perceptron.

```python
import mxnet as mx
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
```

We can get the names of all the free variables by calling `list_arguments`:

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

As can be seen, we name a variable either by its operator's name if it is atomic
(e.g. `Variable`) or by the `opname_varname` convention, where `opname` is the
operator's name and `varname` is assigned by the operator. The `varname`
often means what this variable is for:

- `weight` : the weight parameters
- `bias` : the bias parameters
- `output` : the output
- `label` : input label

On the above example, now we know that there are 6 variables for free
variables. Four of them are learnable parameters, `fc1_weight`, `fc1_bias`,
`fc2_weight`, and `fc2_bias`. These parameters are often initialized by
`mx.initializer` and updated by `mx.optimizer`. The rest two
are for input data: `data` for examples and `softmax_label` for the
according labels. Then it is the iterator's job to feed data into these two
variables.

### Data iterator

An iterator in _MXNet_ should

1. return a data batch or raise a `StopIteration` exception if reaching the end
   when call `next()` in python 2 or `__next()__` in python 3
2. has `reset()` method to restart reading from the beginning
3. has `provide_data` and `provide_label` attributes, the former returns a list
   of `(str, tuple)` pairs, each pair stores an input data variable name and its
   shape. It is similar for `provide_label`, which provides information about
   input labels.

## Reading data in memory
When data is stored in memory backed by either an `NDArray` or ``numpy`` `ndarray`, we can use the [__`NDArrayIter`__](http://mxnet.io/api/python/io.html#mxnet.io.NDArrayIter) to read data as below:


```python
import numpy as np
data = np.random.rand(100,3)
label = np.random.randint(0, 10, (100,))
data_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=30)
for batch in data_iter:
    print([batch.data, batch.label, batch.pad])
```

    [[<NDArray 30x3 @cpu(0)>], [<NDArray 30 @cpu(0)>], 0]
    [[<NDArray 30x3 @cpu(0)>], [<NDArray 30 @cpu(0)>], 0]
    [[<NDArray 30x3 @cpu(0)>], [<NDArray 30 @cpu(0)>], 0]
    [[<NDArray 30x3 @cpu(0)>], [<NDArray 30 @cpu(0)>], 20L]



## Reading Data from CSV Files
*MXNet* provides [`CSVIter`](http://mxnet.io/api/python/io.html#mxnet.io.CSVIter) to read from CSV Files and can be used as below:

There is an iterator called `CSVIter` to read data batches from CSV files. We
first dump `data` into a csv file, and then load the data.

```python
#lets save `data` into a csv file first and try reading it back
np.savetxt('data.csv', data, delimiter=',')
data_iter = mx.io.CSVIter(data_csv='data.csv', data_shape=(3,), batch_size=30)
for batch in data_iter:
    print([batch.data, batch.pad])
```

    [[<NDArray 30x3 @cpu(0)>], 0]
    [[<NDArray 30x3 @cpu(0)>], 0]
    [[<NDArray 30x3 @cpu(0)>], 0]
    [[<NDArray 30x3 @cpu(0)>], 20]



## Custom  Iterator
When the in-built iterators do not suit your application, you can create a custom data iterator.

An iterator in _MXNet_ should  
1. Implement `next()` in ``Python2`` or `__next()__` in ``Python3``,   
   returning `DataBatch` or raise `StopIteration` exception at the end of the datastream.  
2. Implement `reset()` method to restart reading from the beginning.   
3. Have `provide_data` attribute, returning a list of `DataDesc` objects, 
   described [here](http://mxnet.io/api/python/io.html#mxnet.io.DataBatch).  
4. Have `provide_label` attribute, returns similar to `provide_label` information about input labels.  

You can either create a iterator from scratch by defining  `DataBatch` and the iterator(an example is shown below) or reuse existing iterators to create a new iterator. For example, in the image caption application, the input example is an image while the label is a sentence. The we can create a new Iterator by:
- creating image_iter using `ImageRecordIter` which provides multithreaded pre-fetch and augmentation.
- creating caption_iter using `NDArrayIter` or bucketing iterator provided in the rnn package.
- `next()` returns the combined result of `image_iter.next()` and `caption_iter.next()`

Sometimes the provided iterators are not enough for some application. There are
mainly two ways to develop a new iterator. One is creating from scratch: the
following codes define an iterator that creates a given number of data batches
through a data generator `data_gen`.

```python
import numpy as np

class SimpleBatch(mx.io.DataBatch):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
        
class SimpleIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return SimpleBatch(data, label)
        else:
            raise StopIteration
```

We can use the `SimpleIter` to train a simple MLP program below:


```python
import mxnet as mx
num_classes = 10
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
print(net.list_arguments())
print(net.list_outputs())
```

    ['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'softmax_label']
    ['softmax_output']



Here as you see there are 4 variables that are learnable parameters: the *weights* and *biases* of FullyConnected layers *fc1* and *fc2*, two variables for input data: *data* for the training examples and *softmax_label* contains their respective labels and the *softmax_output*. 

The *data* variables are called __free__ variables in the MXNet Symbol land that needs to bound with data.  
Learn more about [Symbol](http://mxnet.io/tutorials/basic/symbol.html).  

We will feed the data iterator into the training problem using the `module` API.  
Learn more about [Module](http://mxnet.io/tutorials/basic/module.html).


```python
import logging
logging.basicConfig(level=logging.INFO)

n = 32
data_iter = SimpleIter(['data'], [(n, 100)], 
                  [lambda s: np.random.uniform(-1, 1, s)],
                  ['softmax_label'], [(n,)], 
                  [lambda s: np.random.randint(0, num_classes, s)])

mod = mx.mod.Module(symbol=net)
mod.fit(data_iter, num_epoch=5)
```

    INFO:root:Epoch[0] Train-accuracy=0.096875
    INFO:root:Epoch[0] Time cost=0.281
    INFO:root:Epoch[1] Train-accuracy=0.087500
    INFO:root:Epoch[1] Time cost=0.008
    INFO:root:Epoch[2] Train-accuracy=0.100000
    INFO:root:Epoch[2] Time cost=0.008
    INFO:root:Epoch[3] Train-accuracy=0.109375
    INFO:root:Epoch[3] Time cost=0.008
    INFO:root:Epoch[4] Train-accuracy=0.103125
    INFO:root:Epoch[4] Time cost=0.012



## Record IO
Record IO is a file format used by *MXNet* for data IO. It compactly packs the data for efficient read and writes from various filesystems including distributed file systems like Hadoop HDFS and AWS S3.
You can learn more about the thinking behind the design of `RecordIO` [here](http://mxnet.io/architecture/note_data_loading.html).

*MXNet* provides __`MXRecordIO`__ and __`MXIndexedRecordIO`__ for sequential access of data and random access of the data.

### MXRecordIO
First let's look at an example on how to read and write sequentially using `mx.recordio.MXRecordIO`. The files are named with a `.rec` extension.


```python
record = mx.recordio.MXRecordIO('tmp.rec', 'w')
for i in range(5):
    record.write('record_%d'%i)
record.close()
```

We can read the data back by opening the file with a option `r` as below:


```python
record = mx.recordio.MXRecordIO('tmp.rec', 'r')
while True:
    item = record.read()
    if not item:
        break
    print (item)
record.close()
```

    record_0
    record_1
    record_2
    record_3
    record_4



### MXIndexedRecordIO
`MXIndexedRecordIO` supports random or indexed access to the data. We will create a indexed record file and a corresponding index file as below:


```python
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
for i in range(5):
    record.write_idx(i, 'record_%d'%i)
record.close()
```

Now we can access the individual records using the keys


```python
record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
record.read_idx(3)
```




    'record_3'



You can also list all the keys in the file.


```python
record.keys
```




    [0, 1, 2, 3, 4]




### Packing and Unpacking data

Each record in a .rec file can contain arbitrary binary data, however most deeplearning tasks need data in a  label/data format.  
`mx.recordio` package provides a few utility functions for such operations, namely: `pack`, `unpack`, `pack_img`, and `unpack_img`.

#### Packing/Unpacking Binary Data.

[__`pack`__](http://mxnet.io/api/python/io.html#mxnet.recordio.pack) and [__`unpack`__](http://mxnet.io/api/python/io.html#mxnet.recordio.unpack) are used for storing float (or 1d array of float) label and binary data:


```python
# `pack`
data = 'data'
label1 = 1.0
header1 = mx.recordio.IRHeader(flag=0, label=label1, id=1, id2=0)
s1 = mx.recordio.pack(header1, data)
print('float label:', repr(s1))
label2 = [1.0, 2.0, 3.0]
header2 = mx.recordio.IRHeader(flag=0, label=label2, id=2, id2=0)
s2 = mx.recordio.pack(header2, data)
print('array label:', repr(s2))
```

    ('float label:', "'\\x00\\x00\\x00\\x00\\x00\\x00\\x80?\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00data'")
    ('array label:', "'\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x80?\\x00\\x00\\x00@\\x00\\x00@@data'")



```python
# `unpack`
print(mx.recordio.unpack(s1))
print(mx.recordio.unpack(s2))
```

    (HEADER(flag=0, label=1.0, id=1, id2=0), 'data')
    (HEADER(flag=3, label=array([ 1.,  2.,  3.], dtype=float32), id=2, id2=0), 'data')



#### Packing/Unpacking Image Data.

*MXNet* provides [__`pack_img`__](http://mxnet.io/api/python/io.html#mxnet.recordio.pack_img) and [__`unpack_img`__](http://mxnet.io/api/python/io.html#mxnet.recordio.unpack_img) to pack/unpack image data. 
Records packed by `pack_img` can be loaded by `mx.io.ImageRecordIter`.


```python
data = np.ones((3,3,1), dtype=np.uint8)
label = 1.0
header = mx.recordio.IRHeader(flag=0, label=label, id=0, id2=0)
s = mx.recordio.pack_img(header, data, quality=100, img_fmt='.jpg')
print(repr(s))
```

    '\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x01\xeb\xff\xd9'



```python
# unpack_img
print(mx.recordio.unpack_img(s))
```

    (HEADER(flag=0, label=1.0, id=0, id2=0), array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]], dtype=uint8))



#### Using tools/im2rec.py
You can also convert raw images into *RecordIO* format using the ``im2rec.py`` utility script that is provided in the *MXNet* [src/tools]() folder.  
An example how to use the script for converting to *RecordIO* format is shown in the `Image IO` section below.


## Image IO

In this section we will learn how to preprocess and load image data in *MXNet*.

There are 4 ways of loading image data in *MXNet*.
   1. Using [__mx.image.imdecode__](http://mxnet.io/api/python/io.html#mxnet.image.imdecode) to load raw image files.
   2. Using [__`mx.img.ImageIter`__](http://mxnet.io/api/python/io.html#mxnet.image.ImageIter) implemented in ``Python`` which is very flexible to customization. It can read from .rec(`RecordIO`) files and raw image files.
   3. Using [__`mx.io.ImageRecordIter`__](http://mxnet.io/api/python/io.html#mxnet.io.ImageRecordIter) implemented on the MXNet backend in C++. This is less flexible to customization but provides various language bindings.  
   4. Creating a Custom Iterator inheriting `mx.io.DataIter`


First set the environment variable `MXNET_HOME` to the root of the *MXNet* source folder


```python
# change this to your mxnet location
MXNET_HOME = '/scratch/mxnet'
```


### Preprocessing Images
Images can be preprocessed in may different ways, some of them are listed below:
- Using `mx.io.ImageRecordIter` which is fast but not very flexible. It is great for simple tasks like image recognition but won't work for more complex tasks like detection and segmentation.
- Using `mx.recordio.unpack_img` (or `cv2.imread`, `skimage`, etc) + `numpy` is flexible but slow due to ``Python`` Glolbal Interpreter Lock(GIL).  
- Using *MXNet* provided `mx.image` package. It stores images in [__`NDArray`__](http://mxnet.io/tutorials/basic/ndarray.html) format and leverages MXNet's [dependency engine](http://mxnet.io/architecture/note_engine.html) to automatically parallelize processing and circumvent GIL.

We will show below some of the frequently used preprocessing routines provided by the `mx.image` package.

Let's download sample images that we can work with.


```python
os.system('wget http://data.mxnet.io/data/test_images.tar.gz')
os.system('tar -xf test_images.tar.gz')
```




#### Loading raw images
Using `mx.image.imdecode` let us first load the images, `imdecode` provides a similar interface to ``OpenCV``.  
**Note: ** You will still need ``OpenCV``(not the CV2 Python library) installed to use `mx.image.imdecode`.


```python
import cv2
import time
tic = time.time()
N = 1000
for i in range(N):
    img = mx.image.imdecode(open('test_images/ILSVRC2012_val_00000001.JPEG').read())
mx.nd.waitall()
print(N/(time.time()-tic), 'images decoded per second with mx.image')
plt.imshow(img.asnumpy()); plt.show()
```

    (210.5357948407611, 'images decoded per second with mx.image')



![](https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/basic/data/output_42_1.png)



#### Image Transformations


```python
# resize to w x h
tmp = mx.image.imresize(img, 100, 70)
plt.imshow(tmp.asnumpy()); plt.show()
```


![](https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/basic/data/output_44_0.png)



```python
# crop a random w x h region from image
tmp, coord = mx.image.random_crop(img, (150, 200))
print(coord)
plt.imshow(tmp.asnumpy()); plt.show()
```

    (80, 128, 150, 200)



![](https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/basic/data/output_45_1.png)



### Loading Data using Image Iterators

Before we see how to read data using the two built-in Image Iterators, lets get a sample dataset __Caltech 101__ dataset that contains 101 classes of objects and convert them into record io format.  
Download and unzip


```python
os.system('wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz -P data/')
os.chdir('data')
os.system('tar -xf 101_ObjectCategories.tar.gz')
os.chdir('../')
```

Let's take a look at the data. As you can see, under the [root folder](./data/101_ObjectCategories) every category has a [subfolder](./data/101_ObjectCategories/yin_yang).

Now let's convert them into record io format using the `im2rec.py` utility scipt. 
First we need to make a list that contains all the image files and their categories:


```python
os.system('python %s/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 --test-ratio=0.2 data/caltech data/101_ObjectCategories'%MXNET_HOME)
```



The resulting [list file](./data/caltech_train.lst) is in the format `index\t(one or more label)\tpath`. In this case there is only one label for each image but you can modify the list to add in more for multi label training.

Then we can use this list to create our record io file


```python
os.system("python %s/tools/im2rec.py --num-thread=4 --pass-through=1 data/caltech data/101_ObjectCategories"%MXNET_HOME)
```



The record io files are now saved at [here](./data)

#### Using ImageIter
[__ImageIter__](http://mxnet.io/api/python/io.html#mxnet.io.ImageIter) is a flexible interface that supports loading of images from both in RecordIO and Raw format.


```python
data_iter = mx.image.ImageIter(batch_size=4, data_shape=(3, 227, 227), 
                              path_imgrec="./data/caltech.rec", 
                              path_imgidx="./data/caltech.idx" )
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()
```

    loading recordio...



![](https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/basic/data/output_55_1.png)


#### Using ImageRecordIter
[__`ImageRecordIter`__](http://mxnet.io/api/python/io.html#mxnet.io.ImageRecordIter) can be used for loading image data saved in record io format. To use ImageRecordIter, simply create an instance by loading your record file:


```python
data_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/caltech.rec", # the target record file
    data_shape=(3, 227, 227), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4, # number of samples per batch
    resize=256 # resize the shorter edge to 256 before cropping
    # ... you can add more augumentation options here. use help(mx.io.ImageRecordIter) to see all possible choices
    )
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()
```


![](https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/basic/data/output_57_0.png)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->

