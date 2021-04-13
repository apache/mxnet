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

This tutorial shows how to install MXNet v1.6 with Jetson support and use it to deploy a pre-trained MXNet model for image classification on a Jetson module.

## What's in this tutorial?

This tutorial shows how to:

1. Install MXNet v1.6 along with its dependencies on a Jetson module (This tutorial has been tested on Jetson Xavier AGX and Jetson Nano modules)

2. Deploy a pre-trained MXNet model for image classifcation on the module

## Who's this tutorial for?

This tutorial would benefit developers working on Jetson modules implementing deep learning applications. It assumes that readers have a Jetson module setup with Jetpack installed, are familiar with the Jetson working environment and are somewhat familiar with deep learning using MXNet.

## Prerequisites

To complete this tutorial, you need:

* A [Jetson module](https://developer.nvidia.com/embedded/develop/hardware) setup with [Jetpack 4.4](https://docs.nvidia.com/jetson/jetpack/release-notes/) installed using NVIDIA [SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager)

* An SSH connection to the module OR display and keyboard setup to directly open shell on the module

* [Swapfile](https://help.ubuntu.com/community/SwapFaq) installed, especially on Jetson Nano for additional memory (increase memory if the inference script terminates with a `Killed` message)

## Installing MXNet v1.6 with Jetson support

To install MXNet with Jetson support, you can follow the [installation guide](https://mxnet.apache.org/get_started/jetson_setup) on MXNet official website.

Alternatively, you can also directly install MXNet v1.6 wheel with Jetson support, hosted on a public s3 bucket. Here are the steps to install this wheel:

*WARNING: this MXNet wheel is provided for your convenience but it contains packages that are not provided nor endorsed by the Apache Software Foundation.
As such, they might contain software components with more restrictive licenses than the Apache License and you'll need to decide whether they are appropriate for your usage. Like all Apache Releases, the
official Apache MXNet (incubating) releases consist of source code only and are found at https://mxnet.apache.org/get_started/download .*

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

```{.python .input}
from mxnet import gluon
import mxnet as mx

# set context
ctx = mx.gpu()

# load pre-trained model
net = gluon.model_zoo.vision.resnet50_v1(pretrained=True, ctx=ctx)
net.hybridize(static_alloc=True, static_shape=True)

# load labels
lbl_path = gluon.utils.download('http://data.mxnet.io/models/imagenet/synset.txt')
with open(lbl_path, 'r') as f:
    labels = [l.rstrip() for l in f]

# download and format image as (batch, RGB, width, height)
img_path = gluon.utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
img = mx.image.imread(img_path)
img = mx.image.imresize(img, 224, 224) # resize
img = mx.image.color_normalize(img.astype(dtype='float32')/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]),
                               std=mx.nd.array([0.229, 0.224, 0.225])) # normalize
img = img.transpose((2, 0, 1)) # channel first
img = img.expand_dims(axis=0) # batchify
img = img.as_in_context(ctx)

prob = net(img).softmax() # predict and normalize output
idx = prob.topk(k=5)[0] # get top 5 result
for i in idx:
    i = int(i.asscalar())
    print('With prob = %.5f, it contains %s' % (prob[0,i].asscalar(), labels[i]))
```

After running the above script, you should get the following output showing the five classes that the image most relates to with probability:
```bash
With prob = 0.41940, it contains n02119789 kit fox, Vulpes macrotis
With prob = 0.28096, it contains n02119022 red fox, Vulpes vulpes
With prob = 0.06857, it contains n02124075 Egyptian cat
With prob = 0.03046, it contains n02120505 grey fox, gray fox, Urocyon cinereoargenteus
With prob = 0.02770, it contains n02441942 weasel
```
