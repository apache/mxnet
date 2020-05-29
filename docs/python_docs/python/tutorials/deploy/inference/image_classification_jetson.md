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

# Image Classication using pretrained ResNet-50 model on Jetson module

This tutorial shows how to install latest MXNet v1.6 with Jetson support and use it to deploy a pre-trained MXNet model for image classification on a Jetson module.

## What's in this tutorial?

This tutorial shows how to:

1. Install MXNet v1.6 with Jetson support along with its dependencies

2. Deploy a pre-trained MXNet model for image classifcation on a Jetson module

### Who's this tutorial for?

This tutorial would benefit developers working on any Jetson module implementing a deep learning application. It assumes that readers have a Jetson module setup, are familiar with the Jetson working environment and are somewhat familiar with deep learning using MXNet.

### How to use this tutorial?

To follow this tutorial, you need to setup a [Jetson module](https://developer.nvidia.com/embedded/develop/hardware) and install latest [Jetpack 4.4](https://docs.nvidia.com/jetson/jetpack/release-notes/) using NVIDIA [SDK manager](https://developer.nvidia.com/nvidia-sdk-manager).

All instructions described in this tutorial can be executed on the any Jetson module directly or via SSH.

## Prerequisites

To complete this tutorial, you will need:

* A Jetson module with Jetpack 4.4 installed
* [Swapfile](https://help.ubuntu.com/community/SwapFaq) installed (in case of Jetson Nano) for additional memory

## Installing MXNet v1.6 with Jetson support

We start by installing MXNet dependencies
```bash
sudo apt-get update
sudo apt-get install -y git build-essential libopenblas-dev libopencv-dev python3-pip
sudo pip3 install -U pip
```

Then we download and install MXNet v1.6 wheel with Jetson support
```bash
wget https://mxnet-public.s3.us-east-2.amazonaws.com/install/jetson/1.6.0/mxnet_cu102-1.6.0-py2.py3-none-linux_aarch64.whl
sudo pip3 install mxnet_cu102-1.6.0-py2.py3-none-linux_aarch64.whl
```

And we are done. You can test the installation now by importing mxnet from python3
```bash
>>> python3 -c 'import mxnet'
```

## Running a pre-trained ResNet-50 model on Jetson

We are now ready to run a pre-trained model and run inference on a Jetson module. In this tutorial we are using ResNet-50 model trained on Imagenet dataset. We run the following classification script with either cpu/gpu context using python3.

```python
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
import urllib.request
import cv2

# set context
ctx = mx.gpu()
dtype = 'float32'
bsize = 1

# download model files
path = 'http://data.mxnet.io/models/imagenet/'
symbol,_ = urllib.request.urlretrieve(path+'resnet/50-layers/resnet-50-symbol.json')
params,_ = urllib.request.urlretrieve(path+'resnet/50-layers/resnet-50-0000.params')
label_file,_ = urllib.request.urlretrieve(path+'synset.txt')

# load model
input_names = ['data', 'softmax_label']
net = nn.SymbolBlock.imports(symbol, input_names, params, ctx)
net.cast(dtype)
net.hybridize(static_alloc=True, static_shape=True)

# load labels
with open(label_file, 'r') as f:
    labels = [l.rstrip() for l in f]

# load image
img_file,_ = urllib.request.urlretrieve('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
img = cv2.imread(img_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224,))
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)

# format input
batch = mx.nd.zeros((bsize,) + img.shape)
for i in range(bsize):
    batch[i] = img
inputs = batch.astype(dtype)
mx_img = [mx.nd.array(inputs,ctx), mx.nd.zeros((bsize,),ctx)]

# infer
results = net(*mx_img)
prob = results[0].asnumpy()
prob = np.squeeze(prob)
a = np.argsort(prob)[::-1]
for i in a[0:5]:
    print('probability=%f, class=%s' %(prob[i], labels[i]))
```

After running the above script, you should get the following output showing the five classes that the image most relates to with probability:
```bash
probability=0.418679, class=n02119789 kit fox, Vulpes macrotis
probability=0.293494, class=n02119022 red fox, Vulpes vulpes
probability=0.029321, class=n02120505 grey fox, gray fox, Urocyon cinereoargenteus
probability=0.026230, class=n02124075 Egyptian cat
probability=0.022557, class=n02085620 Chihuahua
```