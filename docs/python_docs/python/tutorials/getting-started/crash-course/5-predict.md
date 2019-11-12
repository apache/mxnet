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

# Predict with a pre-trained model

A saved model can be used in multiple places, such as to continue training, to fine tune the model, and for prediction. In this tutorial we will discuss how to predict new examples using a pretrained model.

## Prerequisites

Please run the [previous tutorial](4-train.html) to train the network and save its parameters to file. You will need this file to run the following steps.

```{.python .input  n=1}
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from IPython import display
import matplotlib.pyplot as plt
```

To start, we will copy a simple model's definition.

```{.python .input  n=2}
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10))
```

In the last section, we saved all parameters into a file, now let's load it back.

```{.python .input  n=3}
net.load_parameters('net.params')
```

## Predict

Remember the data transformation we did for training? Now we need the same transformation for predicting.

```{.python .input  n=4}
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])
```

Now let's try to predict the first six images in the validation dataset and store the predictions into `preds`.

```{.python .input  n=5}
mnist_valid = datasets.FashionMNIST(train=False)
X, y = mnist_valid[:10]
preds = []
for x in X:
    x = transformer(x).expand_dims(axis=0)
    pred = net(x).argmax(axis=1)
    preds.append(pred.astype('int32').asscalar())
```

Finally, we visualize the images and compare the prediction with the ground truth.

```{.python .input  n=15}
_, figs = plt.subplots(1, 10, figsize=(15, 15))
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
display.set_matplotlib_formats('svg')
for f,x,yi,pyi in zip(figs, X, y, preds):
    f.imshow(x.reshape((28,28)).asnumpy())
    ax = f.axes
    ax.set_title(text_labels[yi]+'\n'+text_labels[pyi])
    ax.title.set_fontsize(14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## Predict with models from Gluon model zoo


The LeNet trained on FashionMNIST is a good example to start with, but too simple to predict real-life pictures. Instead of training large-scale model from scratch, [Gluon model zoo](https://mxnet.apache.org/api/python/gluon/model_zoo.html) provides multiple pre-trained powerful models. For example, we can download and load a pre-trained ResNet-50 V2 model that was trained on the ImageNet dataset.

```{.python .input  n=7}
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download
from mxnet import image

net = models.resnet50_v2(pretrained=True)
```

We also download and load the text labels for each class.

```{.python .input  n=8}
url = 'http://data.mxnet.io/models/imagenet/synset.txt'
fname = download(url)
with open(fname, 'r') as f:
    text_labels = [' '.join(l.split()[1:]) for l in f]
```

We randomly pick a dog image from Wikipedia as a test image, download and read it.

```{.python .input  n=9}
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/\
Golden_Retriever_medium-to-light-coat.jpg/\
365px-Golden_Retriever_medium-to-light-coat.jpg'
fname = download(url)
x = image.imread(fname)
```

Following the conventional way of preprocessing ImageNet data:

1. Resize the short edge into 256 pixes,
2. And then perform a center crop to obtain a 224-by-224 image.

```{.python .input  n=10}
x = image.resize_short(x, 256)
x, _ = image.center_crop(x, (224,224))
plt.imshow(x.asnumpy())
plt.show()
```

Now you may know it is a golden retriever (You can also infer it from the image URL).

The futher data transformation is similar to FashionMNIST except that we subtract the RGB means and divide by the corresponding variances to normalize each color channel.

```{.python .input  n=11}
def transform(data):
    data = data.transpose((2,0,1)).expand_dims(axis=0)
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std
```

Now we can recognize the object in the image now. We perform an additional softmax on the output to obtain probability scores. And then print the top-5 recognized objects.

```{.python .input  n=12}
prob = net(transform(x)).softmax()
idx = prob.topk(k=5)[0]
for i in idx:
    i = int(i.asscalar())
    print('With prob = %.5f, it contains %s' % (
        prob[0,i].asscalar(), text_labels[i]))
```

As can be seen, the model is fairly confident the image contains a golden retriever.
