# Predict with pre-trained models

This tutorial explains how to recognize objects in an image with a pre-trained model, and how to perform feature extraction.

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/install/index.html)

- [Matplotlib](https://matplotlib.org/) and [Jupyter Notebook](http://jupyter.org/index.html).

```
$ pip install matplotlib
```

## Loading

We first download a pre-trained ResNet 18 model that is trained on the ImageNet dataset with over 1 million images and one thousand classes. A pre-trained model contains two parts, a json file containing the model definition and a binary file containing the parameters. In addition, there may be a `synset.txt` text file for the labels.

```python
import mxnet as mx
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
 mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]
```

Next, we load the downloaded model. 

```python
# set the context on CPU, switch to GPU if there is one available
ctx = mx.cpu()
```

```python
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]
```

## Predicting

We first define helper functions for downloading an image and performing the
prediction:

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=False):
    # download and show the image. Remove query string from the file name.
    fname = mx.test_utils.download(url, fname=url.split('/')[-1].split('?')[0])
    img = mx.image.imread(fname)
    if img is None:
        return None
    if show:
        plt.imshow(img.asnumpy())
        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify
    img = img.astype('float32') # for gpu context
    return img

def predict(url):
    img = get_image(url, show=True)
    # compute the predict probabilities
    mod.forward(Batch([img]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))
```

Now, we can perform prediction with any downloadable URL:

```python
predict('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
```

`probability=0.249607, class=n02119022 red fox, Vulpes vulpes` <!--notebook-skip-line-->

`probability=0.172868, class=n02119789 kit fox, Vulpes macrotis` <!--notebook-skip-line-->

![](https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true) <!--notebook-skip-line-->

```python
predict('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true')
```

`probability=0.873920, class=n02110958 pug, pug-dog` <!--notebook-skip-line-->

`probability=0.102659, class=n02108422 bull mastiff` <!--notebook-skip-line-->

![](https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true) <!--notebook-skip-line-->

## Feature extraction

By feature extraction, we mean presenting the input images by the output of an internal layer rather than the last softmax layer. These outputs, which can be viewed as the feature of the raw input image, can then be used by other applications such as object detection.

We can use the ``get_internals`` method to get all internal layers from a Symbol.

```python
# list the last 10 layers
all_layers = sym.get_internals()
all_layers.list_outputs()[-10:]
```

```
['bn1_moving_var',
 'bn1_output',
 'relu1_output',
 'pool1_output',
 'flatten0_output',
 'fc1_weight',
 'fc1_bias',
 'fc1_output',
 'softmax_label',
 'softmax_output']
 ```

An often used layer for feature extraction is the one before the last fully connected layer. For ResNet, and also Inception, it is the flattened layer with name `flatten0` which reshapes the 4-D convolutional layer output into 2-D for the fully connected layer. The following source code extracts a new Symbol which outputs the flattened layer and creates a model.

```python
fe_sym = all_layers['flatten0_output']
fe_mod = mx.mod.Module(symbol=fe_sym, context=ctx, label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
fe_mod.set_params(arg_params, aux_params)
```

We can now invoke `forward` to obtain the features:

```python
img = get_image('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
fe_mod.forward(Batch([img]))
features = fe_mod.get_outputs()[0]
print('Shape',features.shape)
print(features.asnumpy())
assert features.shape == (1, 512)
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
