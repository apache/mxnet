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

# Gluon Fit API

In this tutorial, we will see how to use the [Gluon Fit API](https://cwiki.apache.org/confluence/display/MXNET/Gluon+Fit+API+-+Tech+Design) which is a simple and flexible way to train deep learning models using the [Gluon APIs](http://mxnet.incubator.apache.org/versions/master/gluon/index.html) in Apache MXNet. 

Prior to Fit API, training using Gluon required one to write a custom ["Gluon training loop"](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/logistic_regression_explained.html#defining-and-training-the-model). Fit API reduces the complexity and amount of boiler plate code required to train a model, provides an easy to use and a powerful API. 

To demonstrate the Fit API, this tutorial will train an Image Classification model using the [AlexNet](https://arxiv.org/abs/1404.5997) architecture for the neural network. The model will be trained using the [Fashion-MNIST dataset](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/). 


## Prerequisites

To complete this tutorial, you will need:

- [MXNet](https://mxnet.incubator.apache.org/install/#overview) (The version of MXNet will be >= 1.5.0)
- [GluonCV](https://gluon-cv.mxnet.io)

This tutorial works with both Python 2 and Python 3.



```python
import mxnet as mx
from mxnet import gluon, autograd
from gluoncv import utils
from gluoncv.model_zoo import get_model
from mxnet.gluon.estimator import estimator

ctx = mx.gpu(0) # Or mx.cpu(0) is using a GPU backed machine
mx.random.seed(7) # Set a fixed seed
```

## Dataset

[Fashion-MNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset consists of fashion items divided into ten categories : t-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. 

- It has 60,000 gray scale images of size 28 * 28 for training.  
- It has 10,000 gray scale images os size 28 * 28 for testing/validation. 

We will use ```gluon.data.vision``` package to directly import the Fashion-MNIST dataset and perform pre-processing on it.


```python
# Get the training data 
fashion_mnist_train = gluon.data.vision.FashionMNIST(train=True)

# Get the validation data
fashion_mnist_val = gluon.data.vision.FashionMNIST(train=False)
```

## Exploring the Data


```python
text_labels = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# Let's print the size of the dataset for train and validation.
print ("Number of training samples : %d" % (len(fashion_mnist_train)))
print ("Number of validation samples : %d" % (len(fashion_mnist_val)))


train_first_elem, train_first_label = fashion_mnist_train[0]
print ("Shape of each iamge : ", train_first_elem.shape)
```

    Number of training samples : 60000 <!--notebook-skip-line-->
    Number of validation samples : 10000 <!--notebook-skip-line-->
    Shape of each iamge :  (28, 28, 1) <!--notebook-skip-line-->


Now let's try to visualize the dataset before we proceed further


```python
from IPython import display
import matplotlib.pyplot as plt

# Function to display the data
def show_fashion_mnist_data(images, labels):
    display.set_matplotlib_formats('svg')
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    
    for figure, x, y in zip(figs, images, labels):
        figure.imshow(x.reshape((28, 28)).asnumpy())
        axes = figure.axes
        axes.set_title(text_labels[int(y)])
        axes.title.set_fontsize(12)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    
    plt.show()
```


```python
images, labels = fashion_mnist_train[0:10]
show_fashion_mnist_data(images, labels)
```


![png](https://raw.githubusercontent.com/piyushghai/web-data/master/mxnet/doc/tutorials/gluon/fashion_mnist.png)<!--notebook-skip-line-->


## Pre-processing the data

In order to prepare our data to training the model, we will perform a few pre-processing steps on the dataset. We will :
- Resize the image
- Convert the pixel values in the image from (0 to 255) to (0 to 1)
- Normalize the images with mean 0 and variance 1. 

We will be using ```gluon.data.vision.tranforms``` which provides out of the box transformation APIs.
To read more about the available transformations, check out [the official documentation](https://mxnet.incubator.apache.org/api/python/gluon/data.html#vision-transforms).


```python
transformers = [gluon.data.vision.transforms.Resize(224), # We pick 224 as the model we use takes an input of size 224.
                gluon.data.vision.transforms.ToTensor(), 
                gluon.data.vision.transforms.Normalize(mean = 0, std = 1)]

# Now we will stack all these together.
transform = gluon.data.vision.transforms.Compose(transformers)
```


```python
# Apply the transformations
fashion_mnist_train = fashion_mnist_train.transform_first(transform)
fashion_mnist_val = fashion_mnist_val.transform_first(transform)
```

## Data Loaders

In order to feed the data to our model, we need to use a [Data Loader](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataloader#mxnet.gluon.data.DataLoader). 

Data Loaders are used to read the dataset, create mini-batches from the dataset and feed them to the neural network for training.


```python
batch_size = 256 # Batch size of the images
num_workers = 4 # The number of parallel workers for loading the data using Data Loaders.

train_iter = gluon.data.DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_iter = gluon.data.DataLoader(fashion_mnist_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

## AlexNet Model

AlexNet architecture rose to prominence when it won the [2012 ImageNet LSVRC-2012 competition](http://image-net.org/challenges/LSVRC/2012/).

It contains 5 convolutional layers and 3 fully connected layers. Relu is applied after very convolutional and fully connected layer. Dropout is applied before the first and the second fully connected year.

The [Gluon CV Model Zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html#imagenet) contains a rich collection of state-of-the-art pre-trained models for Computer Vision related tasks.

We will use the ```get_model()``` API from Gluon CV Model Zoo to load the network architecture. 


```python
alexnet_model = get_model('alexnet', pretrained=False, classes = 10, ctx=ctx)
```

## Initialize the model parameters


```python
alexnet_model.initialize(force_reinit=True, init = mx.init.Xavier(), ctx=ctx)
```

## Loss Function, Trainer and Optimizer

After defining the model, let's setup the trainer object for training. 

We will be using ```SoftmaxCrossEntropyLoss``` as the loss function since this is a multi-class classification problem. We will be using ```sgd``` (Stochastic Gradient Descent) as the optimizer. 


```python
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.04 # You can experiment with your own learning rate here

trainer = gluon.Trainer(alexnet_model.collect_params(), 
                        'sgd', {'learning_rate': learning_rate})
```

## Metrics to Capture

We will be most interested in monitoring the accuracy here. Let's define the metrics for that.


```python
train_acc = mx.metric.Accuracy()
```

## Training using Fit API

As stated earlier, Fit API greatly simplifies the boiler plate code and complexity for training using MXNet Gluon.
In just 2 lines of code, we will set up our model for training.


```python
# Define the estimator, by passing to it the model, loss function, metrics, trainer object and context
estimator = estimator.Estimator(net=alexnet_model, 
                                loss=loss_fn, 
                                metrics=train_acc, 
                                trainers=trainer, 
                                context=ctx)

# Magic line
estimator.fit(train_data=train_iter, 
              epochs=5, 
              batch_size=batch_size)
```

    [Epoch 0] [Step 256/60000] time/step: 1.171s accuracy: 0.1133 softmaxcrossentropyloss0: 2.3021 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->
    [Epoch 0] finished in 16.741s: train_accuracy: 0.5996 train_softmaxcrossentropyloss0: 1.0864 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->
    [Epoch 1] finished in 15.313s: train_accuracy: 0.7980 train_softmaxcrossentropyloss0: 0.5410 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->
    [Epoch 2] finished in 15.626s: train_accuracy: 0.8375 train_softmaxcrossentropyloss0: 0.4408 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->   
    [Epoch 3] finished in 15.340s: train_accuracy: 0.8575 train_softmaxcrossentropyloss0: 0.3893 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->
    [Epoch 4] finished in 15.420s: train_accuracy: 0.8694 train_softmaxcrossentropyloss0: 0.3560 <!--notebook-skip-line-->


## Comparison with Trainer Loop (Older way)

Without the Fit API, the code to train using the Gluon Trainer Loop looks something like this below :


```python
epochs = 5

alexnet_model = get_model('alexnet', pretrained=False, classes = 10, ctx=ctx)
alexnet_model.initialize(force_reinit=True, init = mx.init.Xavier(), ctx=ctx)

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.04 # You can experiment with your own learning rate here

trainer = gluon.Trainer(alexnet_model.collect_params(), 
                        'sgd', {'learning_rate': learning_rate})

acc = mx.metric.Accuracy()

# Gluon Training loop goes here 

for epoch in range(epochs):
    train_acc = 0.0
    train_loss = 0.0
    validation_acc = 0.0
    acc.reset()
    for data, label in train_iter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = alexnet_model(data)
            loss_val = loss_fn(output, label)
            
        loss_val.backward()
        trainer.step(batch_size)
        acc.update(preds = output, labels=label)
        l = loss_val.mean().asscalar()
        train_loss += l
        train_acc += acc.get()[1]
    
    print("Epoch %d: , train acc %.3f , train loss %.3f " % (epoch, train_acc/len(train_iter), train_loss/ len(train_iter)))

# Gluon Training loop ends
```

    Epoch 0: , train acc 0.412 , train loss 1.106 <!--notebook-skip-line-->
    Epoch 1: , train acc 0.777 , train loss 0.543 <!--notebook-skip-line-->
    Epoch 2: , train acc 0.832 , train loss 0.439 <!--notebook-skip-line-->
    Epoch 3: , train acc 0.857 , train loss 0.387 <!--notebook-skip-line-->
    Epoch 4: , train acc 0.867 , train loss 0.357 <!--notebook-skip-line-->


The training loop involves : 
- Manually iterating over epochs and batches
- Recording the gradients during the forward pass
- Computing the loss function
- Calling the back propagation on the loss function
- Applying the training step, i.e, updating the weights
- Recording any useful metrics in the meanwhile

## Summary

In this tutorial, we learnt how to use ```Gluon Fit APIs``` for training a deep learning model and compared it with the existing gluon trainer loop. 

## Next Steps 

- To learn more about deep learning with MXNet Gluon, see [Deep Learning - The Straight Dope](https://gluon.mxnet.io)
- For more hands on learning about deep learning, checkout out [Dive into Deep Learning](https://d2l.ai)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->