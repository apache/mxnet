# Methods of applying data augmentation (Module API)

Data Augmentation is a regularization technique that's used to avoid overfitting when training Machine Learning models. Although the technique can be applied in a variety of domains, it's very common in Computer Vision. Adjustments are made to the original images in the training dataset before being used in training. Some example adjustments include translating, cropping, scaling, rotating, changing brightness and contrast. We do this to reduce the dependence of the model on spurious characteristics; e.g. training data may only contain faces that fill 1/4 of the image, so the model trained without data augmentation might unhelpfully learn that faces can only be of this size.

In this tutorial we discuss the different interfaces available in MXNet to perform data augmentation if you're using the Module API. We start by showing a complete example using Module's [`ImageIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imageiter#mxnet.image.ImageIter), and then unpack the example to gain a greater understanding of the internals. In the process you'll learn about augmentation functions, [`mxnet.image.Augmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=augmen#mxnet.image.Augmenter) classes and Augmenter lists.


```python
%matplotlib inline
import mxnet as mx # used version '1.0.0' at time of writing
import numpy as np
from matplotlib.pyplot import imshow
import multiprocessing
import os

mx.random.seed(42) # set seed for repeatability
```

We define a utility function below, that will be used for visualising the augmentations in the tutorial.


```python
def plot_mx_array(array):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    assert array.shape[2] == 3, "RGB Channel should be last"
    imshow((array.clip(0, 255)/255).asnumpy())
```

```python
image_dir = os.path.join("data", "images")
mx.test_utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/inputs/0.jpg', dirname=image_dir)
example_image = mx.image.imread(os.path.join(image_dir,"0.jpg")).astype("float32")
plot_mx_array(example_image)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/outputs/use/output_5_0.png)<!--notebook-skip-line-->


## Quick start using [`ImageIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imageiter#mxnet.image.ImageIter)

One of the most convenient ways to augment your image data is via arguments of [`mxnet.image.ImageIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imageiter#mxnet.image.ImageIter), but you'll need to reference the documentation of [`mxnet.image.CreateAugmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=createaugmenter#mxnet.image.CreateAugmenter) to see a full list of available options. Under the hood, additional arguments passed to [`ImageIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imageiter#mxnet.image.ImageIter) are collected as keyword arguments (`**kwargs`), and are passed to [`mxnet.image.CreateAugmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=createaugmenter#mxnet.image.CreateAugmenter). We'll see this in more detail in the sections below, but [`mxnet.image.CreateAugmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=createaugmenter#mxnet.image.CreateAugmenter) creates a list of  [`mxnet.image.Augmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=augmen#mxnet.image.Augmenter)s corresponding to each type of augmentation (e.g. crop, flip, change of brightness, etc.), and this list will be iterated though and the augmentations applied in turn. Alternatively, you can create this list yourself and pass it to [`ImageIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imageiter#mxnet.image.ImageIter) via the `aug_list` argument.


We show a simple example of this below, after creating an `images.lst` file used by the [`ImageIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imageiter#mxnet.image.ImageIter). Use [`tools/im2rec.py`](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) to create the `images.lst` if you don't already have this for your data.

```python
path_to_image = os.path.join("images", "0.jpg")
index = 0
label = 0.
list_file_content = "{0}\t{1:.5f}\t{2}".format(index, label, path_to_image)

path_list_file = os.path.join(image_dir, "images.lst")
with open(path_list_file, 'w') as f:
    f.write(list_file_content)

```

```python
training_iter = mx.image.ImageIter(batch_size = 1,
                                   data_shape = (3, 300, 300),
                                   path_root= 'data', path_imglist=path_list_file,
                                   rand_crop=0.5, rand_mirror=True, inter_method=10,
                                   brightness=0.125, contrast=0.125, saturation=0.125,
                                   pca_noise=0.02
                                   )
```


```python
for batch in training_iter:
    assert batch.data[0].shape == (1, 3, 300, 300)
    assert batch.label[0].shape == (1,)
    sample = batch.data[0][0].transpose(axes=[1,2,0])
    plot_mx_array(sample)
    break
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/outputs/use/output_28_1.png)<!--notebook-skip-line-->

[`mxnet.image.ImageDetIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imagedetiter#mxnet.image.ImageDetIter) works similarly (with [`mxnet.image.CreateDetAugmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=createdetaugmenter#mxnet.image.CreateDetAugmenter)), but [`mxnet.io.ImageRecordIter`](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imagerecorditer#mxnet.io.ImageRecordIter) has a slightly different interface, so reference the documentation [here](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imagerecorditer#mxnet.io.ImageRecordIter) if you're using Record IO data format.

## Manual Augmentation

Although the vast majority of cases will be covered using the augmentation arguments of [`mxnet.image.ImageIter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=imageiter#mxnet.image.ImageIter) as we've seen above, sometime you'll want more fine grained control of augmentations. We will now dive into some of the lower level methods for image augmentation, that you can use to manually apply augmentations to images.

### Augmentation Functions

MXNet provides a small number of augmentation functions that are quick and easy to use, but they are limited to positional augmentations (such as [`mxnet.image.random_crop`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=random_crop#mxnet.image.random_crop) and [`mxnet.image.resize_short`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=random_crop#mxnet.image.resize_short) functions) as opposed to color augmentations (such as brightness jitter). Although these functions are easy to apply, the [`mxnet.image.Augmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=augmen#mxnet.image.Augmenter) classes are much more comprehensive and just as easy to use, as we'll see in the next section.


```python
aug_image, crop_box = mx.image.random_crop(example_image, size=(100, 100))
plot_mx_array(aug_image)
assert aug_image.shape == (100, 100, 3)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/outputs/use/output_16_0.png)<!--notebook-skip-line-->


### Augmenter Classes

You can apply a wide variety of positional and color augmentations with [`mxnet.image.Augmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=augmen#mxnet.image.Augmenter) classes, and using them is the recommended approach for applying augmentations manually. After creating an instance of an Augmenter with the required parameters, you can call the Augmenter just as you would a function. Under the hood a `__call__` method is defined which applies the augmentation. Augmenters with randomness are randomized each time the Augmenter is called, so calling the same Augmenter twice will give different results on the same input.


```python
aug = mx.image.RandomCropAug(size=(100, 100))
aug_image = aug(example_image)
plot_mx_array(aug_image)
assert aug_image.shape == (100, 100, 3)
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/outputs/use/output_19_0.png)<!--notebook-skip-line-->


### Augmenter list

Very often you'll want to apply many different types of augmentation to an image. Instead of nesting the calls of Augmenters, a natural structure for handling a large number of Augmenters is a list. You can construct this list manually, or you can use helper functions like [`mxnet.image.CreateAugmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=createaugmenter#mxnet.image.CreateAugmenter) to create these lists automatically.

Object detection tasks require the same positional augmentations to be applied to the data and the label, so you should use [`mxnet.image.CreateDetAugmenter`](https://mxnet.incubator.apache.org/api/python/image/image.html?highlight=createdetaugmenter#mxnet.image.CreateDetAugmenter) which handles this case.


```python
# created manually
aug_list = [mx.image.RandomCropAug(size=(100, 100)), mx.image.BrightnessJitterAug(brightness=1)]
aug_image = example_image.copy()
for aug in aug_list:
    aug_image = aug(aug_image)
plot_mx_array(aug_image)
assert all([isinstance(a, mx.image.Augmenter) for a in aug_list])
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/outputs/use/output_22_1.png)<!--notebook-skip-line-->



```python
# created automatically
aug_list = mx.image.CreateAugmenter(data_shape=(3, 300, 300), rand_crop=0.5,
        rand_mirror=True, mean=True, brightness=0.125, contrast=0.125,
        saturation=0.125, pca_noise=0.05, inter_method=10)
aug_image = example_image.copy()
for aug in aug_list:
    aug_image = aug(aug_image)
plot_mx_array(aug_image)
assert all([isinstance(a, mx.image.Augmenter) for a in aug_list])
```


![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/outputs/use/output_23_1.png)<!--notebook-skip-line-->


__*Watch Out!*__ Check some examples that are output after applying all the augmentations. You may find that the augmentation steps are too severe and may actually prevent the model from learning. Some of the augmentation parameters used in this tutorial are set high for demonstration purposes (e.g. `brightness=1`); you might want to reduce them if your training error stays too high during training. Some examples of excessive augmentation are shown below:

<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/data_aug/outputs/use//severe_aug.png" alt="Drawing" style="width: 700px;"/>

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->