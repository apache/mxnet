# Gluon: from experiment to deployment, an end to end example

## Overview

MXNet Gluon API comes with a lot of great features and it can provide you everything you need from experiment to deploy the model.
In this tutorial, we will walk you through a common used case on how to build a model using gluon, train it on your data, and deploy it for inference.

Let's say you want to build a service that provides flower species recognition. A common use case is, you don't have enough data to train a good model like ResNet50.
What you can do is utilize pre-trained model from Gluon, tweak the model according to your neeed, fine-tune the model on your small dataset, and deploy the model to integrate with your service.

We will use the [Oxford 102 Category Flower Dateset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) as an example to show you the steps.

## Prepare training data

You can use this [script](https://github.com/Arsey/keras-transfer-learning-for-oxford102/blob/master/bootstrap.py) to download and organize your data into train, test, and validation sets. Simply run:
```python
python bootstrap.py
```

Now your data will be organized into the following format, all the images belong to the same category will be put together
```
data
├── train
│   ├── 0
│   │   ├── image_06736.jpg
│   │   ├── image_06741.jpg
...
│   ├── 1
│   │   ├── image_06755.jpg
│   │   ├── image_06899.jpg
...
├── test
│   ├── 0
│   │   ├── image_00731.jpg
│   │   ├── image_0002.jpg
...
│   ├── 1
│   │   ├── image_00036.jpg
│   │   ├── image_05011.jpg

```

## Define Hyper-paramerters
Now let's first import neccesarry packages:
```python
import mxnet as mx
import numpy as np
import os, time

from mxnet import gluon, init
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
```

and define the hyper parameter we will use for fine-tuning:
```python
classes = 102

epochs = 1
lr = 0.001
per_device_batch_size = 32
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, np.inf]

num_gpus = 0
num_workers = 1
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)
```

## Data pre-processing

We can use Gluon DataSet API, DataLoader API, and Transform API to load the images and do data augmentation:
```python
jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


path = './data'
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'valid')
test_path = os.path.join(path, 'test')

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)
 ```


## Loading pre-trained model

We will use pre-trained ResNet50_v2 model, all you need to do is re-define the last softmax layer for your case. Specify the number of classes in your data and initialize the weights.
You can also add layers to the network according to your needs.

Before we go to training, one important part is to hybridize your model, it will convert your imperative code to mxnet symbolic graph. It's much more efficient to train a symbolic model,
and you can also serialize and save the network archietecure and parameters for inference.

```python
model_name = 'ResNet50_v2'
finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Fine-tuning model on your custom dataset

Now let's define the test metrics and start fine-tuning.

```python
def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()


lr_counter = 0
num_batch = len(train_data)

for epoch in range(epochs):
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate*lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    _, val_acc = test(finetune_net, val_data, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, time.time() - tic))

_, test_acc = test(finetune_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))
```

Note we are able to reach a test accuracy of 93% with only 20 epochs in less than 20 minutes, this is really fast because we used the
pre-trained weights from ResNet50, it's been trained on a marge larget dataset: ImageNet, so it works really well to capture features on our small dataset.

## Save fine-tuned model

We now have a trained our custom model. This can be exported into files using the export function. The export function will export the model architecture into a .json file and model parameters into a .params file.

```python
net.export("flower-recognition", epoch=1)
```
export in this case creates flower-recognitio-symbol.json and flower-recognitio-0001.params in the current directory.

## Load and inference using C API

(WIP)

## References

https://gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html
