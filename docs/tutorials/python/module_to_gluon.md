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

# Converting Module API code to the Gluon API

Sometimes you find yourself in the situation where the model you want to use has been written using the symbolic Module API rather than the simpler, easier-to-debug, more flexible, imperative Gluon API. In this tutorial, we will give you a comprehensive guide for transforming Module code to Gluon code.

The different steps to take into consideration are:

I) Data loading

II) Model definition

III) Loss

IV) Training Loop

V) Exporting Models

VI) Loading Models for Inference

In the following section we will look at 1:1 mappings between the Module and the Gluon ways of training a neural network.

## I - Data Loading

In this section we will be looking at the difference in loading data between Module and Gluon.
Let's first import a few Python modules.

```python
from collections import namedtuple
import logging
logging.basicConfig(level=logging.INFO)
import random

import numpy as np
import mxnet as mx
from mxnet.gluon.data import ArrayDataset, DataLoader
from mxnet.gluon import nn
from mxnet import gluon

# parameters
batch_size = 5
dataset_length = 50

# random seeds
random.seed(1)
np.random.seed(1)
mx.random.seed(1)

```

#### Module

When using the Module API we use a [`DataIter`](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=dataiter#mxnet.io.DataIter), in addition to the data itself, the [`DataIter`](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=dataiter#mxnet.io.DataIter) contains information about the name of the input symbols. 

In the Module API, `DataIter`s are responsible for both holding the data and iterating through it. Some `DataIter`s support multi-threading like the [`ImageRecordIter`](https://mxnet.incubator.apache.org/api/python/io/io.html#mxnet.io.ImageRecordIter), while other don't, such as the [`NDArrayIter`](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=ndarrayiter#mxnet.io.NDArrayIter).

Let's create some random data, following the same format as grayscale 28x28 images.


```python
train_data = np.random.rand(dataset_length, 28,28).astype('float32')
train_label = np.random.randint(0, 10, (dataset_length,)).astype('float32')
```

We can now wraps this data into an ArrayIterator that will create batches of data using the first dimension of the provided array as the batch dimension. 

```python
data_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size, shuffle=False, data_name='data', label_name='softmax_label')
for batch in data_iter:
    print(batch.data[0].shape, batch.label[0])
    break
```

    (5, 28, 28) 
    [5. 0. 3. 4. 9.]
    <NDArray 5 @cpu(0)>


#### Gluon

With Gluon, the preferred method is to use a [`DataLoader`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataloader#mxnet.gluon.data.DataLoader) that makes use of a [`Dataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataset#mxnet.gluon.data.Dataset) to asynchronously prefetch the data. 

The Gluon API offers you the ability to efficiently fetch data and separate the concerns of loading versus holding data. The DataLoader role is to request certain indices of the dataset. The Dataset has ownership of the data.
The `Dataset` data can be in or out of memory, and the `DataLoader` role is to request certain indices of the dataset, in the main thread or through multi-processing (or multi-threaded) workers and batch the data together. 

```python
dataset = ArrayDataset(train_data, train_label)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
for data, label in dataloader:
    print(data.shape, label)
    break
```

    (5, 28, 28) 
    [5. 0. 3. 4. 9.]
    <NDArray 5 @cpu(0)>

You can check the [`Dataset` and `DataLoader` tutorials](https://mxnet.incubator.apache.org/tutorials/gluon/datasets.html) out. You can either rewrite your code in order to use one of the provided [`Dataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataset#mxnet.gluon.data.Dataset) class, like the [`ArrayDataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=arraydataset#mxnet.gluon.data.ArrayDataset) or the [`ImageFolderDataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=imagefolderdataset#mxnet.gluon.data.vision.datasets.ImageFolderDataset)


## II - Model Definition

Let's look at the model definition from the [MNIST Module Tutorial](https://mxnet.incubator.apache.org/tutorials/python/mnist.html):

#### Module

For the Module API, you define the data flow by setting `data` keyword argument of one layer to the next.
You then bind the symbolic model to a specific compute context and specify the symbol names for the data and the label.

```python

# context
ctx = mx.cpu()

def get_module_network():
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)
    fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")
    fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
    mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return mlp

mlp = get_module_network()
# Bind model to Module
mlp_model = mx.mod.Module(symbol=mlp, context=ctx, data_names=['data'], label_names=['softmax_label'])
```

#### Gluon

In Gluon, for the equivalent model, you would create a `Sequential` block, in that case a `HybridSequential` block to allow for future hybridization since we are only using [hybridizable blocks](https://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html). The flow of the data will be automatically set from one layer to the next, since they are held in a `Sequential` block.
Note that we don't need named symbols for the input, and we show how the loss is handled in Gluon in the next section.

```python
def get_gluon_network():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.Flatten(),
            nn.Dense(units=128, activation="relu"),
            nn.Dense(units=64, activation="relu"),
            nn.Dense(units=10)
        )
    return net

net = get_gluon_network()
```

## III - Loss

The loss, that you are trying to minimize using an optimization algorithm like SGD, is defined differently in the Module API than in Gluon.


#### Module


In the module API, the loss is part of the network. It has usually a forward pass result, that is the inference value, and a backward pass that is the gradient of the output with respect to that particular loss.

For example, the [sym.SoftmaxOutput](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html?highlight=softmaxout#mxnet.symbol.SoftmaxOutput) is a softmax output in the forward pass and the gradient with respect to the cross-entropy loss in the backward pass.

```python
# Softmax with cross entropy loss, directly part of the network
out = mx.sym.SoftmaxOutput(data=mlp, name='softmax')
```

#### Gluon


In Gluon, it is a lot more transparent. Losses, like the [SoftmaxCrossEntropyLoss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html?highlight=softmaxcross#mxnet.gluon.loss.SoftmaxCrossEntropyLoss), are only computing the actual value of the loss. You then call `.backward()` on the loss value to compute the gradient of the parameters with respect to that loss. At inference time, you simply call `.softmax()` on your output to get the output of your network normalized according to the softmax function.


```python
# We simply create a loss function we will use in our training loop
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
```

In the next section we will show how you use this loss function in Gluon to generate the loss value in the main training loop.

## IV - Training Loop


#### Module

The Module API provides a [`.fit()`](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=.fit#mxnet.module.BaseModule.fit) function that takes care of fitting training data to your symbolic model. With Gluon, your execution flow controls the data flow, so you need to write your own loop. It might seems like it is more verbose, but you have a lot more control as to what is happening during the training. 
With the [`.fit()`](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=.fit#mxnet.module.BaseModule.fit) function, you control the metric reporting, checkpointing or weights initialization through a lot of different keyword arguments (check the [docs](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=.fit#mxnet.module.BaseModule.fit)). That is where you define the optimizer for example.

```python
mlp_model.fit(data_iter,  # train data
              eval_data=data_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              force_init=True,
              force_rebind=True,
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              num_epoch=5)  # train for 5 full dataset passes
```

```INFO:root:Epoch[4] Train-accuracy=0.070000```<!--notebook-skip-line-->

```INFO:root:Epoch[4] Time cost=0.038```<!--notebook-skip-line-->

```INFO:root:Epoch[4] Validation-accuracy=0.125000```<!--notebook-skip-line-->

#### Gluon


With Gluon, you do these operations directly in the training loop, and the optimizer is part of the [`Trainer`](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=trainer#mxnet.gluon.Trainer) object that handles the weight updates of your parameters.

Notice the `loss.backward()` we call before updating the weight as mentionned in the previous section

```python
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) # Initialize network and trainer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

metric = mx.metric.Accuracy() # Pick a metric

for e in range(5): # start of epoch
    
    for data, label in dataloader: # start of mini-batch
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        with mx.autograd.record():
            output = net(data) # forward pass
            loss = loss_fn(output, label) # get loss
            
        loss.backward() # compute gradients
        trainer.step(data.shape[0]) # update weights with SGD
        metric.update(label, output) # update the metrics # end of mini-batch

    name, acc = metric.get()
    print('training metrics at epoch %d: %s=%f'%(e, name, acc))
    metric.reset() # end of epoch
```

```training metrics at epoch 3: accuracy=0.155000```<!--notebook-skip-line-->

```training metrics at epoch 4: accuracy=0.145000```<!--notebook-skip-line-->

The Gluon training code is more verbose than the simple `.fit` from Module. However that is also the main advantage, there is no black magic going on here, you have full control of your training loop. You can for example easily set breakpoints, modify a learning rate or print data during the training flow. This flexibility also makes easy to implement more complex use-case like gradient accumulation across batches.

## V - Exporting Model

The ultimate purpose of training a model is to be able to export it and share it, whether it is for deployment or simply reproducibility purposes.

#### Module


With the Module API, you can save model using the [`.save_checkpoint()`](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=save_chec#mxnet.module.Module.save_checkpoint) and get a `-symbol.json` and a `.params` file that represent your network. 


```python
mlp_model.save_checkpoint('module-model', epoch=5)
# module-model-0005.params module-model-symbol.json
```

```INFO:root:Saved checkpoint to "module-model-0005.params"```<!--notebook-skip-line-->

#### Gluon



With Gluon, network parameters are associated with a `Block`, but the execution flow is controlled in python through the code in `.forward()` function. Hence only [hybridized networks]() can be exported with a `-symbol.json` and `.params` file using [`.export()`](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=export#mxnet.gluon.HybridBlock.export), non-hybridized models can only have their parameters exported using [`.save_parameters()`](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=save_pa#mxnet.gluon.Block.save_parameters). Check this great tutorial to learn more: [Saving and Loading Gluon Models](https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html).


Any models:

```python
# save only the parameters
net.save_parameters('gluon-model.params')
# gluon-model.params
```

Hybridized models:

```python
# save the parameters and the symbolic representation
net.hybridize()
net(mx.nd.ones((1,1,28,28), ctx))

net.export('gluon-model-hybrid', epoch=5)
# gluon-model-hybrid-symbol.json gluon-model-hybrid-0005.params
```

## VI - Loading Model for Inference


#### Module


For inference, in the Module API, you need to first load the parameters and symbol, bind the symbol to a module and load the corresponding parameters. You can then pass a batch of data through that module and request the output of the network.


```python
# Load the symbol and parameters
sym, arg_params, aux_params = mx.model.load_checkpoint('module-model', 5)

# Bind them in a module
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,1,28,28))], 
         label_shapes=mod._label_shapes)

# Set the parameters
mod.set_params(arg_params, aux_params, allow_missing=True)

# Run the inference
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([mx.nd.ones((1,28,28))]))
prob = mod.get_outputs()[0].asnumpy()
print("Output probabilities: {}".format(prob))
```

`Output probabilities: [[0.05537598 0.03889056 0.06126577 0.08879893 0.12371024 0.05759033 0.1378248  0.26134694 0.07905186 0.09614458]]`<!--notebook-skip-line-->

#### Gluon (Symbolic Model)

For the Gluon API, it is a lot simpler. You can just load a serialized model in a [`SymbolBlock`](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=symbolblo#mxnet.gluon.SymbolBlock) and run inference directly.

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    net = gluon.SymbolBlock.imports('module-model-symbol.json', ['data', 'softmax_label'], 'module-model-0005.params')
prob = net(mx.nd.ones((1,1,28,28)), mx.nd.ones(1)) # note the second argument here to account for the softmax_label symbol
print("Output probabilities: {}".format(prob.asnumpy()))
```

`Output probabilities: [[0.05537598 0.03889056 0.06126577 0.08879893 0.12371024 0.05759033 0.1378248  0.26134694 0.07905186 0.09614458]]`<!--notebook-skip-line-->

#### Gluon (Imperative Model)

```python
net = get_gluon_network()
net.load_parameters('gluon-model.params')
prob = net(mx.nd.ones((1,1,28,28))).softmax()
print("Output probabilities: {}".format(prob.asnumpy()))
```

`Output probabilities: [[0.01298077 0.00173413 0.01661885 0.3362421  0.00536332 0.02099853 0.01413316 0.5528366  0.0133819  0.02571066]]`<!--notebook-skip-line-->

## Conclusion

This tutorial lead you through the steps necessary to train a deep learning model and showed you the differences between the symbolic approach of the Module API and the imperative one of the Gluon API. If you need more help converting your Module API code to the Gluon API, reach out to the community on the [discuss forum](https://discuss.mxnet.io)!
You can also compare the scripts for training MNIST in [Gluon](https://mxnet.incubator.apache.org/tutorials/gluon/mnist.html) and [Module](https://mxnet.incubator.apache.org/tutorials/python/mnist.html).



<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
