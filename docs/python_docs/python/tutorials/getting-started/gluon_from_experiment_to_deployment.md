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


# Gluon: from experiment to deployment

## Overview
MXNet Gluon API comes with a lot of great features, and it can provide you everything you need: from experimentation to deploying the model. In this tutorial, we will walk you through a common use case on how to build a model using gluon, train it on your data, and deploy it for inference.

Let's say you need to build a service that provides flower species recognition. A common problem is that you don't have enough data to train a good model. In such cases, a technique called Transfer Learning can be used to make a more robust model.
In Transfer Learning we make use of a pre-trained model that solves a related task, and was trained on a very large standard dataset, such as ImageNet. ImageNet is from a different domain, but we can utilize the knowledge in this pre-trained model to perform the new task at hand.

Gluon provides State of the Art models for many of the standard tasks such as Classification, Object Detection, Segmentation, etc. In this tutorial we will use the pre-trained model [ResNet50 V2](https://arxiv.org/abs/1603.05027) trained on ImageNet dataset. This model achieves 77.11% top-1 accuracy on ImageNet. We seek to transfer as much knowledge as possible for our task of recognizing different species of flowers.




## Prerequisites

To complete this tutorial, you need:

- [Build MXNet from source](https://mxnet.apache.org/get_started/build_from_source) with Python(Gluon) and C++ Packages
- Learn the basics about Gluon with [A 60-minute Gluon Crash Course](https://gluon-crash-course.mxnet.io/)


## The Data

We will use the [Oxford 102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) as an example to show you the steps.
We have prepared a utility file to help you download and organize your data into train, test, and validation sets. Run the following Python code to download and prepare the data:


```{.python .input}
import mxnet as mx
data_util_file = "oxford_102_flower_dataset.py"
base_url = "https://raw.githubusercontent.com/apache/mxnet/master/docs/tutorial_utils/data/{}?raw=true"
mx.test_utils.download(base_url.format(data_util_file), fname=data_util_file)
import oxford_102_flower_dataset

# download and move data to train, test, valid folders
path = './data'
oxford_102_flower_dataset.get_data(path)
```

Now your data will be organized into train, test, and validation sets, images belong to the same class are moved to the same folder.

## Training using Gluon

### Define Hyper-parameters

Now let's first import necessary packages:


```{.python .input}
import math
import os
import time

from mxnet import autograd
from mxnet import gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo.vision import resnet50_v2
```

Next, we define the hyper-parameters that we will use for fine-tuning. We will use the [MXNet learning rate scheduler](../packages/gluon/training/learning_rates/learning_rate_schedules.ipynb) to adjust learning rates during training.
Here we set the `epochs` to 1 for quick demonstration, please change to 40 for actual training.

```{.python .input}
classes = 102
epochs = 1
lr = 0.001
per_device_batch_size = 32
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
# learning rate change at following epochs
lr_epochs = [10, 20, 30]

num_gpus = mx.device.num_gpus()
# you can replace num_workers with the number of cores on you device
num_workers = 8
device = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)
```

Now we will apply data augmentations on training images. This makes minor alterations on the training images, and our model will consider them as distinct images. This can be very useful for fine-tuning on a relatively small dataset, and it will help improve the model. We can use the Gluon [DataSet API](../../api/gluon/data/index.rst#mxnet.gluon.data.Dataset), [DataLoader API](../../api/gluon/data/index.rst#mxnet.gluon.data.DataLoader), and [Transform API](../../api/gluon/data/index.rst#mxnet.gluon.data.Dataset.transform) to load the images and apply the following data augmentations:
1. Randomly crop the image and resize it to 224x224
2. Randomly flip the image horizontally
3. Randomly jitter color and add noise
4. Transpose the data from `[height, width, num_channels]` to `[num_channels, height, width]`, and map values from [0, 255] to [0, 1]
5. Normalize with the mean and standard deviation from the ImageNet dataset.

For validation and inference, we only need to apply step 1, 4, and 5. We also need to save the mean and standard deviation values for inference using other language bindings.

```{.python .input}
jitter_param = 0.4
lighting_param = 0.1

# mean and std for normalizing image value in range (0,1)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

training_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

validation_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# save mean and std NDArray values for inference
mean_img = mx.np.stack([mx.np.full((224, 224), m) for m in mean])
std_img = mx.np.stack([mx.np.full((224, 224), s) for s in std])
mx.npx.savez('mean_std_224.np', **{"mean_img": mean_img, "std_img": std_img})

train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'valid')
test_path = os.path.join(path, 'test')

# loading the data and apply pre-processing(transforms) on images
train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(training_transformer),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

### Loading pre-trained model


We will use pre-trained ResNet50_v2 model which was pre-trained on the [ImageNet Dataset](http://www.image-net.org/) with 1000 classes. To match the classes in the Flower dataset, we must redefine the last softmax (output) layer to be 102, then initialize the parameters.

Before we go to training, one unique Gluon feature you should be aware of is hybridization. It allows you to convert your imperative code to a static symbolic graph, which is much more efficient to execute. There are two main benefits of hybridizing your model: better performance and easier serialization for deployment. The best part is that it's as simple as just calling `net.hybridize()`. To know more about Gluon hybridization, please follow the [hybridization tutorial](../packages/gluon/blocks/hybridize.rst).



```{.python .input}
# load pre-trained resnet50_v2 from model zoo
finetune_net = resnet50_v2(pretrained=True, device=device)

# change last softmax layer since number of classes are different
finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), device=device)
# hybridize for better performance
finetune_net.hybridize()

num_batch = len(train_data)

# setup learning rate scheduler
iterations_per_epoch = math.ceil(num_batch)
# learning rate change at following steps
lr_steps = [epoch * iterations_per_epoch for epoch in lr_epochs]
schedule = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps, factor=lr_factor, base_lr=lr)

# setup optimizer with learning rate scheduler, metric, and loss function
sgd_optimizer = mx.optimizer.SGD(learning_rate=lr, lr_scheduler=schedule, momentum=momentum, wd=wd)
metric = mx.gluon.metric.Accuracy()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

### Fine-tuning model on your custom dataset

Now let's define the test metrics and start fine-tuning.



```{.python .input}
def test(net, val_data, device):
    metric = mx.gluon.metric.Accuracy()
    for i, (data, label) in enumerate(val_data):
        data = gluon.utils.split_and_load(data, device, even_split=False)
        label = gluon.utils.split_and_load(label, device, even_split=False)
        outputs = [net(x) for x in data]
        metric.update(label, outputs)
    return metric.get()

trainer = gluon.Trainer(finetune_net.collect_params(), optimizer=sgd_optimizer)

# start with epoch 1 for easier learning rate calculation
for epoch in range(1, epochs + 1):

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, (data, label) in enumerate(train_data):
        # get the images and labels
        data = gluon.utils.split_and_load(data, device, even_split=False)
        label = gluon.utils.split_and_load(label, device, even_split=False)
        with autograd.record():
            outputs = [finetune_net(x) for x in data]
            loss = [softmax_cross_entropy(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.mean().item() for l in loss]) / len(loss)
        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch
    _, val_acc = test(finetune_net, val_data, device)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | learning-rate: %.3E | time: %.1f' %
          (epoch, train_acc, train_loss, val_acc, trainer.learning_rate, time.time() - tic))

_, test_acc = test(finetune_net, test_data, device)
print('[Finished] Test-acc: %.3f' % (test_acc))
```

Following is the training result:
```text
[Epoch 40] Train-acc: 0.945, loss: 0.354 | Val-acc: 0.955 | learning-rate: 4.219E-04 | time: 17.8
[Finished] Test-acc: 0.952
```
In the previous example output, we trained the model using an [AWS p3.8xlarge instance](https://aws.amazon.com/ec2/instance-types/p3/) with 4 Tesla V100 GPUs. We were able to reach a test accuracy of 95.5% with 40 epochs in around 12 minutes. This was really fast because our model was pre-trained on a much larger dataset, ImageNet, with around 1.3 million images. It worked really well to capture features on our small dataset.


### Save the fine-tuned model


We now have a trained our custom model. This can be serialized into model files using the export function. The export function will export the model architecture into a `.json` file and model parameters into a `.params` file.



```{.python .input}
finetune_net.export("flower-recognition", epoch=epochs)

```

`export` creates `flower-recognition-symbol.json` and `flower-recognition-0040.params` (`0040` is for 40 epochs we ran) in the current directory. These files can be used for model deployment using the `HybridBlock.import` API.

## What's next

You can find more ways to run inference and deploy your models here:
1. [MXNet Model Server Examples](https://github.com/awslabs/mxnet-model-server/tree/master/examples)

## References

1. [Transfer Learning for Oxford102 Flower Dataset](https://github.com/Arsey/keras-transfer-learning-for-oxford102)
2. [Gluon book on fine-tuning](https://www.d2l.ai/chapter_computer-vision/fine-tuning.html)
3. [Gluon CV transfer learning tutorial](https://cv.gluon.ai/build/examples_classification/transfer_learning_minc.html)
4. [Gluon crash course](https://gluon-crash-course.mxnet.io/)
5. [Gluon CPP inference example](https://github.com/apache/mxnet/blob/master/cpp-package/example/inference/)
