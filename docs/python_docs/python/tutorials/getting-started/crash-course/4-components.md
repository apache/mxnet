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
# Necessary components that are not in the network

The data and algorithms are not the only components that you need to create your
trained deep learning models. In this notebook, you will learn about some of the
common components used for training your own machine learning models. Here is a
list of components that are commonly found in training a model in MXNet.

1. Initialization
2. Loss function
    1. Built-in
    2. Custom
3. Optimizers
4. Metrics

```python
from mxnet import np, npx,gluon
import mxnet as mx
from mxnet.gluon import nn
npx.set_np()

ctx = mx.cpu()
```

## Initialization

In the previous notebook, you used `net.initialize()` to initialize the network
before you could do a forward pass. Now, you will look at initialization in a little
more detail.

First, define the `sequential` network that you used earlier and initialize it.
After you initialize it, print the parameters using `collect_params()` method.

```python
net = nn.Sequential()

net.add(nn.Dense(5,in_units=3,activation='relu'),
        nn.Dense(25, activation='relu'),
        nn.Dense(2)
       )

net
```

```python
net.initialize()
params = net.collect_params()

for key,value in params.items():
    print(key,value)


```

The weights for the first layer and second layer are **-1** since the shape is
deferred to runtime. After the first forward computation/pass, the shape is
inferred from the shape of the data and the parameters are not initialized.
After the first forward pass, the shape and parameters are initialized. Look at 
what the shape looks like after the first forward pass.

```python
x = np.random.uniform(-1,1,(10,3))
net(x)  # Forward computation

params = net.collect_params()
for key,value in params.items():
    print(key,value)


```

#### Built-in Initialization

MXNet includes some common initializers built-in for ease of use. They include
some of the commonly used initialzers like

- Constant
- Normal
- Xavier

For more information, see [Initializers TODO: CHANGE
LINK](https://mxnet.apache.org/versions/1.6/api/python/docs/api/initializer/index.html)

When you used `net.intialize()`, MXNet initializes the weight matrices uniformly
by drawing random values with uniform-distribution between −0.07 and 0.07 and
updates the bias parameters by setting them all to 0.

To initialize your network using different built-in types, you have to use the
`init` keyword argument in the `initialize()` method. Now, look at how to do
this using a `constant` init and a `normal` init.

```python
from mxnet import init

# Constant init initializes the weights to be a constant value for all the params
net.initialize(init=init.Constant(3),ctx=ctx)
print(net[0].weight.data()[0])
```

Normal Init initializes weights with random values sampled from a normal
distribution with a mean of zero and standard deviation of sigma. If you have
already initialized the weight but want to reinitialize the weight, set the
`force_reinit` flag to `True`.

```python
net.initialize(init=init.Normal(sigma=0.2),force_reinit=True,ctx=ctx)
print(net[0].weight.data()[0])
```

## Components used in a training loop

Till now you have seen how to create an algorithm and initialized it using mxnet
APIs as well as learned the basics of using mxnet. When you start training the
ML algorithm, how does it actually learn or train?

There are three main components of what happens during training an algorithm.

1. Loss function which is used to calculate how far the model is from the true
distribution
2. Autograd, the mxnet auto differentiation tool to calculate the gradients to
optimize the parameters
3. Optimizer which is used to update the parameters based on an optimization
algorithm

You have already learned about autograd in the previous notebook. In this
notebook, you will learn about the remaining components as well as dive a little
deeper into initialization.

## Loss function

Loss functions are used to train neural networks and help the algorithm learn
the data distribution. The loss function computes the difference between output
from the neural network and ground truth value. This score is used to update the
neural network weights during training. First, you should look at a simple example.

Suppose you have a neural network `net` and the data is stored in variable
`data`. Take 5 total records and the output from the neural network after
the first epoch is given by the following variable. The values are in millions.

```python
net = gluon.nn.Dense(1)
net.initialize()

nn_input = np.array([1.2,0.56,3.0,0.72,0.89]).reshape(5,1)

nn_output = net(nn_input)
nn_output
```

But the ground truth value of the data is stored in `groundtruth_label` is

```python
groundtruth_label = np.array([[0.0083],
                             [0.00382],
                             [0.02061],
                             [0.00495],
                             [0.00639]]).reshape(5,1)
```

For this problem, use the L2 Loss. L2Loss, also called Mean Squared Error, is a
regression loss function that computes the squared distances between the target
values and the output of the neural network. It is defined as:

$$L = \frac{1}{2N}\sum_i{|label_i − pred_i|)^2}$$

Compared to L1, L2 loss it is a smooth function and it creates larger gradients
for large loss values. However due to the squaring it puts high weight on
outliers.

```python
def L2Loss(output_values, true_values):
    return np.mean((output_values - true_values)**2,axis=1)/2

L2Loss(nn_output,groundtruth_label)
```

Now, do the same thing using the mxnet API

```python
from mxnet.gluon import nn, loss as gloss
loss = gloss.L2Loss()

loss(nn_output,groundtruth_label)
```

A network can improve by iteratively updating its weights to minimise this loss.
Some tasks use a combination of multiple loss functions, but often you’ll just
use one. MXNet Gluon provides a number of the most commonly used loss functions,
and you’ll choose certain loss functions depending on your network and task.
Some common task and loss function pairs include:

- regression: L1Loss, L2Loss

- classification: SigmoidBinaryCrossEntropyLoss, SoftmaxCrossEntropyLoss

- embeddings: HingeLoss

#### Customizing your Loss functions

You can also create custom loss functions using **Loss Blocks**. For more
information see []()

You can inherit the base `Loss` class and write your own `forward` method. The
backward propagation will be automatically computed by autograd. However that
only holds true if you can build your loss from existing operators.

```python
from mxnet.gluon.loss import Loss

class custom_L1_loss(Loss):
    def __init__(self,weight=None, batch_axis=0, **kwargs):
        super(custom_L1_loss,self).__init__(weight, batch_axis, **kwargs)

    def forward(self, pred, label):
        l = np.abs(label - pred)
        l = l.reshape(len(l),)
        return l
    
L1 = custom_L1_loss()
L1(nn_output,groundtruth_label)
```

```python
l1=gloss.L1Loss()
l1(nn_output,groundtruth_label)
```

## Optimizer

The loss function is how much the parameters are changing based on how far the
model is. Optimizer is how the model weights or parameters are updated based on
the loss function. In Gluon, this optimization step is performed by the
`gluon.Trainer`.

Lets look at a basic example of how to call the `gluon.Trainer` method.

```python
from mxnet import optimizer
```

```python
trainer = gluon.Trainer(net.collect_params(),
                       optimizer='Adam',
                       optimizer_params={
                           'learning_rate':0.1,
                           'wd':0.001
                       })
```

When creating a **Gluon Trainer** you must provide the trainer object with
1. A collection of parameters that need to be learnt. The collection of
parameters will the weights and biases of your network that you are training.
2. An Optimization algorithm (optimizer) that you want to use for training. This
algorithm will be used to update the parameters every training iteration when
`trainer.step` is called. For more information, see [optimizers in v1.6 TODO:
CHANGE](https://mxnet.apache.org/versions/1.6/api/python/docs/api/optimizer/index.html)

```python
curr_weight = net.weight.data()
print(curr_weight)
```

```python
batch_size = len(nn_input)
trainer.step(batch_size)
print(net.weight.data())
```

```python
print(curr_weight - net.weight.grad() * 1 / 5)

```

## Metrics

MXNet includes a `metrics` API that you can use to evaluate how your model is
doing. This is typically used during training to monitor performance on the
validation set. MXNet includes commonly used metrics:

-
[Accuracy](https://mxnet.apache.org/versions/2.0/api/python/docs/api/metric/index.html#mxnet.metric.Accuracy)
-
[CrossEntropy](https://mxnet.apache.org/versions/2.0/api/python/docs/api/metric/index.html#mxnet.metric.CrossEntropy)
- [Mean squared
error](https://mxnet.apache.org/versions/1.6/api/python/docs/api/metric/index.html#mxnet.metric.MSE)
- [Root mean squared error
(RMSE)](https://mxnet.apache.org/versions/1.6/api/python/docs/api/metric/index.html#mxnet.metric.RMSE)

Lets look at an binary class classification example.

```python
# Vector of likelihoods for all the classes
pred = np.array([[0.1, 0.9], [0.05, 0.95], [0.83, 0.17],[0.63,0.37]])

labels = np.array([1,1,0,1])
```

Before you can calculate the accuracy of your model, the metric (accuracy)
should be instantiated before the training loop

```python
from mxnet.gluon.metric import Accuracy

acc = Accuracy()
```

To run and calculate the updated accuracy for each batch or epoch, you can call
the `update()` method. This method uses labels and predictions which can be
either class index or vector of likelihoods for all classes.

```python
acc.update(labels=labels, preds = pred)
```

#### Creating custom metrics

In addition to built-in metrics, if you want to create a custom metric, you can
use the following skeleton code. The code inherits from the `EvalMetric` base
class.

```
def custom_metric(EvalMetric):
    def __init__(self):
        super().init()

    def update(self,labels, preds):
        pass

```

Here is an example using the Precision metric. First, define the two values
`labels` and `preds`.

```python
labels = np.array([0,1,1,0,0,1,0,0,1,0,1,1])
preds = np.array([0,1,1,1,1,0,0,1,0,0,0,0])
```

Next, define the custom metric class `precision` and instantiate it

```python
from mxnet.gluon.metric import EvalMetric

class precision(EvalMetric):
    def __init__(self):
        super().__init__(name='Precision')
        
    def update(self,labels, preds):
        tp_labels = (labels == 1)
        true_positives = sum(preds[tp_labels]== 1)
        fp_labels = (labels == 0)
        false_positives = sum(preds[fp_labels]== 1)
        return true_positives / (true_positives + false_positives)
        
p = precision()
```

And finally, call the `update` method to return the precision for your data

```python
p.update(np.array(y_true),np.array(y_pred))
```
