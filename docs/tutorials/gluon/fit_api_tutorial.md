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


# MXNet Gluon Fit API

In this tutorial, we will see how to use the [Gluon Fit API](https://cwiki.apache.org/confluence/display/MXNET/Gluon+Fit+API+-+Tech+Design) which is the easiest way to train deep learning models using the [Gluon API](http://mxnet.incubator.apache.org/versions/master/gluon/index.html) in Apache MXNet. 

With the Fit API, you can train a deep learning model with miminal amount of code. Just specify the network, loss function and the data you want to train on. You don't need to worry about the boiler plate code to loop through the dataset in batches(often called as 'training loop'). Advanced users can still do this for bespolke training loops, but most use cases will be covered by the Fit API.

To demonstrate the Fit API, this tutorial will train an Image Classification model using the [ResNet-18](https://arxiv.org/abs/1512.03385) architecture for the neural network. The model will be trained using the [Fashion-MNIST dataset](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/). 


## Prerequisites

To complete this tutorial, you will need:

- [MXNet](https://mxnet.incubator.apache.org/install/#overview) (The version of MXNet will be >= 1.5.0)
- [Jupyter Notebook](https://jupyter.org/index.html) (For interactively running the provided .ipynb file)




```python
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
from mxnet.gluon.estimator import estimator, event_handler

ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
mx.random.seed(7) # Set a fixed seed
```

## Dataset

[Fashion-MNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset consists of fashion items divided into ten categories: t-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. 

- It has 60,000 gray scale images of size 28 * 28 for training.  
- It has 10,000 gray scale images os size 28 * 28 for testing/validation. 

We will use the ```gluon.data.vision``` package to directly import the Fashion-MNIST dataset and perform pre-processing on it.


```python
# Get the training data 
fashion_mnist_train = gluon.data.vision.FashionMNIST(train=True)

# Get the validation data
fashion_mnist_val = gluon.data.vision.FashionMNIST(train=False)
```


```python
transforms = [gluon.data.vision.transforms.Resize(224), # We pick 224 as the model we use takes an input of size 224.
                gluon.data.vision.transforms.ToTensor()]

# Now we will stack all these together.
transforms = gluon.data.vision.transforms.Compose(transforms)
```


```python
# Apply the transformations
fashion_mnist_train = fashion_mnist_train.transform_first(transforms)
fashion_mnist_val = fashion_mnist_val.transform_first(transforms)
```


```python
batch_size = 256 # Batch size of the images
num_workers = 4 # The number of parallel workers for loading the data using Data Loaders.

train_data_loader = gluon.data.DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_data_loader = gluon.data.DataLoader(fashion_mnist_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

## Model and Optimizers

Let's load the resnet-18 model architecture from [Gluon Model Zoo](http://mxnet.apache.org/api/python/gluon/model_zoo.html) and initialize it's parameters. The Gluon Model Zoo contains a repository of pre-trained models as well the model architecture definitions. We are using the model architecture from the model zoo in order to train it from scratch.


```python
resnet_18_v1 = vision.resnet18_v1(pretrained=False, classes = 10)
resnet_18_v1.initialize(init = mx.init.Xavier(), ctx=ctx)
```

We will be using ```SoftmaxCrossEntropyLoss``` as the loss function since this is a multi-class classification problem. We will be using ```sgd``` (Stochastic Gradient Descent) as the optimizer. You can experiment with a different optimizer as well. 


```python
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
```

Let's define the trainer object for training the model.

```python
learning_rate = 0.04 # You can experiment with your own learning rate here
num_epochs = 2 # You can run training for more epochs
trainer = gluon.Trainer(resnet_18_v1.collect_params(), 
                        'sgd', {'learning_rate': learning_rate})
```

## Train using Fit API

As stated earlier, Fit API greatly simplifies the boiler plate code and complexity for training using MXNet Gluon.

In the basic usage example, with just 2 lines of code, we will set up our model for training.

### Basic Usage


```python
train_acc = mx.metric.Accuracy() # Metric to monitor

# Define the estimator, by passing to it the model, loss function, metrics, trainer object and context
est = estimator.Estimator(net=resnet_18_v1, 
                                loss=loss_fn, 
                                metrics=train_acc, 
                                trainers=trainer, 
                                context=ctx)

# Magic line
est.fit(train_data=train_data_loader,
              epochs=num_epochs, 
              batch_size=batch_size)
```

    [Epoch 0] [Step 256/60000] time/step: 1.420s accuracy: 0.0938 softmaxcrossentropyloss0: 2.9419 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->
    [Epoch 0] finished in 51.375s: train_accuracy: 0.7916 train_softmaxcrossentropyloss0: 0.5750 <!--notebook-skip-line-->
    [Epoch 1] [Step 256/60000] time/step: 0.414s accuracy: 0.8555 softmaxcrossentropyloss0: 0.3621 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->
    [Epoch 1] finished in 49.889s: train_accuracy: 0.8854 train_softmaxcrossentropyloss0: 0.3157 <!--notebook-skip-line-->


### Advanced Usage

Fit API is also customizable with several `Event Handlers` which give a fine grained control over the steps in training and exposes callback methods that provide control over the stages involved in training. Available callback methods are: `train_begin`, `train_end`, `batch_begin`, `batch_end`, `epoch_begin` and `epoch_end`.

One can use built-in event handlers such as `LoggingHandler`, `CheckpointHandler` or `EarlyStoppingHandler` to log and save the model at certain timesteps during training and stopping the training when the model's performance plateaus. One can also create a custom handler by inheriting [`EventHandler`](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/estimator/event_handler.py#L31).


```python
# Let's reset the model, trainer and accuracy objects from above

resnet_18_v1.initialize(force_reinit=True, init = mx.init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(resnet_18_v1.collect_params(), 
                        'sgd', {'learning_rate': learning_rate})
train_acc = mx.metric.Accuracy()

```


```python
# Define the estimator, by passing to it the model, loss function, metrics, trainer object and context
est = estimator.Estimator(net=resnet_18_v1,
                                loss=loss_fn,
                                metrics=train_acc,
                                trainers=trainer, 
                                context=ctx)

# Define the handlers, let's say Checkpointhandler
checkpoint_handler = event_handler.CheckpointHandler(estimator=est,
                                                     filepath='./my_best_model.params',
                                                     monitor='train_accuracy', # Monitors a metric
                                                     save_best_only=True) # Save the best model in terms of 
                                                                         # training accuracy

# Magic line
est.fit(train_data=train_data_loader,
              epochs=num_epochs,
              event_handlers=checkpoint_handler, # Add the event handlers
              batch_size=batch_size)
```

    [Epoch 0] [Step 256/60000] time/step: 0.426s accuracy: 0.1211 softmaxcrossentropyloss0: 2.6261 
    .... <!--notebook-skip-line-->
    [Epoch 0] finished in 50.390s: train_accuracy: 0.7936 train_softmaxcrossentropyloss0: 0.5639 <!--notebook-skip-line-->
    [Epoch 1] [Step 256/60000] time/step: 0.414s accuracy: 0.8984 softmaxcrossentropyloss0: 0.2958 <!--notebook-skip-line-->
    .... <!--notebook-skip-line-->
    [Epoch 1] finished in 50.474s: train_accuracy: 0.8871 train_softmaxcrossentropyloss0: 0.3101 <!--notebook-skip-line-->


You can load the saved model, by using ```load_parameters``` API in Gluon. For more details refer to the [Loading model parameters from file tutorial](http://mxnet.incubator.apache.org/versions/master/tutorials/gluon/save_load_params.html#saving-model-parameters-to-file)

## Summary

In this tutorial, we learnt how to use ```Gluon Fit APIs``` for training a deep learning model and also saw an option to customize it with the use of Event Handlers.
For more references and advanced usage details can be found in the [documentation](http://mxnet.apache.org/api/python/gluon/gluon.html).

## Next Steps 

- To learn more about deep learning with MXNet Gluon, see [Deep Learning - The Straight Dope](https://gluon.mxnet.io)
- For more hands on learning about deep learning, check out [Dive into Deep Learning](https://d2l.ai)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
