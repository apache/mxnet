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

# Object Detection using pretrained GluonCV YOLO model on Jetson module

This tutorial shows how to install MXNet v1.6 (with Jetson support) and GluonCV on a Jetson module and deploy a pre-trained GluonCV model for object detection.

## What's in this tutorial?

This tutorial shows how to:

1. Install MXNet v1.6 along with its dependencies on a Jetson module

2. Install GluonCV and its dependencies on the module

3. Deploy a pre-trained YOLO model for object detection on the module

*Note: This tutorial has been tested on Jetson Xavier AGX  and Jetson TX2 modules.*

## Who's this tutorial for?

This tutorial would benefit developers working on Jetson modules implementing deep learning applications. It assumes that readers have a Jetson module setup with Jetpack installed, are familiar with the Jetson working environment and are somewhat familiar with deep learning using MXNet.

## Prerequisites

To complete this tutorial, you need:

* A [Jetson module](https://developer.nvidia.com/embedded/develop/hardware) setup with [Jetpack 4.4](https://docs.nvidia.com/jetson/jetpack/release-notes/) installed using NVIDIA [SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager)

* Display (needed to view matplotlib plot) and keyboard setup to directly open shell on the module

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
sudo apt-get install -y git build-essential libopenblas-dev python3-pip
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

## Installing GluonCV and its dependencies

We can install GluonCV on Jetson module using the following commands: 
```bash
sudo apt-get update
sudo apt-get install -y python3-scipy python3-pil python3-matplotlib
sudo apt autoremove -y
sudo pip3 install gluoncv
```

## Running a pre-trained GluonCV YOLOv3 model on Jetson

We are now ready to deploy a pre-trained model and run inference on a Jetson module. In this tutorial we are using YOLOv3 model trained on Pascal VOC dataset with Darknet53 as the base model. The object detection script below can be run with either cpu/gpu context using python3.

*Note: If running with GPU context, set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable cuDNN autotune*
```bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
```

Here's the object detection python script:
```python
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx

# set context
ctx = mx.gpu()

# load model
net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)

# load input image
im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='dog.jpg')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
x = x.as_in_context(ctx)

# call forward and show plot
class_IDs, scores, bounding_boxs = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()
```

This is the input image:
![Input](https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg)

After running the above script, you should get the following plot as output:
![Output](https://gluon-cv.mxnet.io/_images/sphx_glr_demo_yolo_001.png)
