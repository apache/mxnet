
# Converting Module API code to the Gluon API

Sometimes, you find yourself in the situation where the model you want to use has been written using the symbolic Module API rather than the imperative Gluon API. In this tutorial, we will give you a comprehensive guide you can use in order to convert a given model to use Gluon.

The different element to take in consideration are:

I) Data loading

II) Model definition

III) Loss

IV) Training Loop

V) Exporting Models

In the following section we will look at 1:1 mapping between the Module and the Gluon ways.

## I - Data Loading


```python
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import mxnet as mx
from mxnet.gluon.data import ArrayDataset, DataLoader
from mxnet.gluon import nn
from mxnet import gluon

batch_size = 5
dataset_length = 200
```

#### Module

When using the Module API we use a [`DataIter`](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=dataiter#mxnet.io.DataIter), in addition to the data itself, the [`DataIter`](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=dataiter#mxnet.io.DataIter) contains information about the name of the input symbols.

Let's create some random data, following the same format as grayscale 28x28 images.


```python
train_data = np.random.rand(dataset_length, 28,28).astype('float32')
train_label = np.random.randint(0, 10, (dataset_length,)).astype('float32')
```


```python
data_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size, shuffle=False, data_name='data', label_name='softmax_label')
for batch in data_iter:
    print(batch.data[0].shape, batch.label[0])
    break;
```

    (5, 28, 28) 
    [5. 0. 3. 4. 9.]
    <NDArray 5 @cpu(0)>


#### Gluon

With Gluon, the preferred method is to use a [`DataLoader`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataloader#mxnet.gluon.data.DataLoader) that make use of a [`Dataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataset#mxnet.gluon.data.Dataset) to prefetch asynchronously the data.


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


#### Notable differences

- Gluon keeps a strict separation between data holding, and data loading / fetching. The `Dataset` role is to hold onto some data, in or out of memory, and the `DataLoader` role is to request certain indices of the dataset, in the main thread or through multi-processing workers. This flexible API allows to efficiently pre-fetch data and separate the concerns. 
- In the module API, `DataIter`s are responsible for both holding the data and iterating through it. Some `DataIter` support multi-threading like the [`ImageRecordIter`](https://mxnet.incubator.apache.org/api/python/io/io.html#mxnet.io.ImageRecordIter), while other don't like the `NDArrayIter`.

You can checkout the [`Dataset` and `DataLoader` tutorial](https://mxnet.incubator.apache.org/tutorials/gluon/datasets.html). You can either rewrite your code in order to use one of the provided [`Dataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=dataset#mxnet.gluon.data.Dataset) class, like the [`ArrayDataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=arraydataset#mxnet.gluon.data.ArrayDataset) or the [`ImageFolderDataset`](https://mxnet.incubator.apache.org/api/python/gluon/data.html?highlight=imagefolderdataset#mxnet.gluon.data.vision.datasets.ImageFolderDataset), or you can simply wrap your existing [`DataIter`](https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=dataiter#mxnet.io.DataIter) to have a similar usage pattern as a `DataLoader`:


```python
class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label

    def next(self):
        return self.__next__() # for Python 2
```


```python
data_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size)
data_iter_loader = DataIterLoader(data_iter)
for data, label in data_iter_loader:
    print(data.shape, label)
    break
```

    (5, 28, 28) 
    [5. 0. 3. 4. 9.]
    <NDArray 5 @cpu(0)>


## II - Model definition

Let's look at the model definition from the [MNIST Module Tutorial](https://mxnet.incubator.apache.org/tutorials/python/mnist.html):


```python
ctx = mx.gpu()
```

#### Module


```python
data = mx.sym.var('data')
data = mx.sym.flatten(data=data)
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# Bind model to Module
mlp_model = mx.mod.Module(symbol=mlp, context=ctx, data_names=['data'], label_names=['softmax_label'])
```

#### Gluon

In Gluon, for a sequential model like that, you would create a `Sequential` block, in that case a `HybridSequential` block to allow for future hybridization since we are only using hybridizable blocks. Learn more [about hybridization](https://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html).


```python
net = nn.HybridSequential()
with net.name_scope():
    net.add(
        nn.Flatten(),
        nn.Dense(units=128, activation="relu"),
        nn.Dense(units=64, activation="relu"),
        nn.Dense(units=10)
    )
```

## III - Loss

The loss, that you are trying to minimize using an optimization algorithm like SGD, is defined differently in the Module API and in Gluon.

In the module API, the loss is part of the network. It has usually a forward result, that is the inference value, and a backward pass that is the gradient of the output with respect to that particular loss.

For example the [sym.SoftmaxOutput](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html?highlight=softmaxout#mxnet.symbol.SoftmaxOutput) is a softmax output in the forward pass and the gradient with respect to the cross-entropy loss in the backward pass.

In Gluon, it is a lot more transparent. Losses, like the [SoftmaxCrossEntropyLoss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html?highlight=softmaxcross#mxnet.gluon.loss.SoftmaxCrossEntropyLoss), are only computing the actual value of the loss. You then call `.backward()` on the loss value to compute the gradient of the parameters with respect to that loss. At inference time, you simply call `.softmax()` on your output to get the output of your network normalized according to the softmax function.

#### Module


```python
# Softmax with cross entropy loss, directly part of the network
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
```

#### Gluon


```python
# We simply create a loss function we will use in our training loop
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
```

## IV - Training Loop

The Module API provides a [`.fit()`](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=.fit#mxnet.module.BaseModule.fit) functions that takes care of fitting training data to your symbolic model. With Gluon, you execution flow controls the data flow, so you need to write your own loop. It might seems like it is more verbose, but you have a lot more control as to what is happening during the training. 
With the [`.fit()`](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=.fit#mxnet.module.BaseModule.fit) function, you control the metric reporting, checkpointing, through a lot of different keyword arguments (check the [docs](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=.fit#mxnet.module.BaseModule.fit)). That is where you define the optimizer for example.

With Gluon, you do these operations directly in the training loop, and the optimizer is part of the [`Trainer`](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=trainer#mxnet.gluon.Trainer) object that handles the weight updates of your parameters.

#### Module


```python
mlp_model.fit(data_iter,  # train data
              eval_data=data_iter,  # validation data
              optimizer='adam',  # use SGD to train
              force_init=True,
              force_rebind=True,
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              num_epoch=5)  # train for at most 10 dataset passes
```

```INFO:root:Epoch[4] Train-accuracy=0.070000```<!--notebook-skip-line-->

```INFO:root:Epoch[4] Time cost=0.038```<!--notebook-skip-line-->

```INFO:root:Epoch[4] Validation-accuracy=0.125000```<!--notebook-skip-line-->

#### Gluon


```python
# Initialize network and trainer
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Pick a metric
metric = mx.metric.Accuracy()

for e in range(5): # start of epoch
    
    for data, label in dataloader: # start of mini-batch
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        with mx.autograd.record():
            output = net(data) # forward pass
            loss = loss_fn(output, label) # get loss
            loss.backward() # compute gradients
        
        trainer.step(data.shape[0]) # update weights with SGD
        metric.update(label, output) # update the metrics
        # end of mini-batch
    name, acc = metric.get()
    print('training metrics at epoch %d: %s=%f'%(e, name, acc))
    metric.reset()
    # end of epoch
```

```training metrics at epoch 3: accuracy=0.155000```<!--notebook-skip-line-->

```training metrics at epoch 4: accuracy=0.145000```<!--notebook-skip-line-->


## V - Exporting model

The ultimate purpose of training a model is to be able to export it and share it, whether it is for deployment or simply reproducibility purposes.

With the Module API, you can save model using the [`.save_checkpoint()`](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=save_chec#mxnet.module.Module.save_checkpoint) and get a `-symbol.json` and a `.params` file that represent your network. 

With Gluon, network parameters are associated with a `Block`, but the execution flow is controlled in python through the code in `.forward()` function. Hence only [hybridized networks]() can be exported with a `-symbol.json` and `.params` file using [`.export()`](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=export#mxnet.gluon.HybridBlock.export), non-hybridized models can only have their parameters exported using [`.save_parameters()`](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=save_pa#mxnet.gluon.Block.save_parameters). Check this great tutorial to learn more: [Saving and Loading Gluon Models](https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html).

#### Module


```python
mlp_model.save_checkpoint('module-model', epoch=5)
# nodule-model-0005.params module-model-symbol.json
```

```INFO:root:Saved checkpoint to "module-model-0005.params"```<!--notebook-skip-line-->

#### Gluon


```python
# save only the parameters
net.save_parameters('gluon-model.params')
# gluon-model.params
```


```python
# save the parameters and the symbolic representation
net.hybridize()
net(mx.nd.ones((1,1,28,28), ctx))

net.export('gluon-model-hybrid', epoch=5)
# gluon-model-hybrid-symbol.json gluon-model-hybrid-0005.params
```

## Conclusion

This tutorial lead you through the steps necessary to train a deep learning model and showed you the differene between the symbolic approach of the Module API and the imperative Gluon API. If you need help converting your Module API code to the Gluon API, reach out to the community on the [discuss forum](https://discuss.mxnet.io)!


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->