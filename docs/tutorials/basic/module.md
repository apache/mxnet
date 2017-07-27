# Module - Neural network training and inference

Training a neural network involves quite a few steps. One need to specify how
to feed input training data, initialize model parameters, perform forward and
backward passes through the network, update weights based on computed gradients, do
model checkpoints, etc. During prediction, one ends up repeating most of these
steps. All this can be quite daunting to both newcomers as well as experienced
developers.

Luckily, MXNet modularizes commonly used code for training and inference in
the `module` (`mod` for short) package. `Module` provides both high-level and
intermediate-level interfaces for executing predefined networks. One can use
both interfaces interchangeably. We will show the usage of both interfaces in
this tutorial.

## Prerequisites

To complete this tutorial, we need:

- MXNet. See the instructions for your operating system in [Setup and Installation](http://mxnet.io/get_started/install.html).  

- [Jupyter Notebook](http://jupyter.org/index.html) and [Python Requests](http://docs.python-requests.org/en/master/) packages.
```
pip install jupyter requests
```

## Preliminary

In this tutorial we will demonstrate `module` usage by training a
[Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP)
on the [UCI letter recognition](https://archive.ics.uci.edu/ml/datasets/letter+recognition)
dataset.

The following code downloads the dataset and creates an 80:20 train:test
split. It also initializes a training data iterator to return a batch of 32
training examples each time. A separate iterator is also created for test data.

```python
import logging
logging.getLogger().setLevel(logging.INFO)
import mxnet as mx
import numpy as np

fname = mx.test_utils.download('http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data')
data = np.genfromtxt(fname, delimiter=',')[:,1:]
label = np.array([ord(l.split(',')[0])-ord('A') for l in open(fname, 'r')])

batch_size = 32
ntrain = int(data.shape[0]*0.8)
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)
```

Next, we define the network.

```python
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
net = mx.sym.SoftmaxOutput(net, name='softmax')
mx.viz.plot_network(net)
```

## Creating a Module

Now we are ready to introduce module. The commonly used module class is
`Module`. We can construct a module by specifying the following parameters:

- `symbol`: the network definition
- `context`: the device (or a list of devices) to use for execution
- `data_names` : the list of input data variable names
- `label_names` : the list of input label variable names

For `net`, we have only one data named `data`, and one label named `softmax_label`,
which is automatically named for us following the name `softmax` we specified for the `SoftmaxOutput` operator.

```python
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])
```

## Intermediate-level Interface

We have created module. Now let us see how to run training and inference using module's intermediate-level APIs. These APIs give developers flexibility to do step-by-step
computation by running `forward` and `backward` passes. It's also useful for debugging.

To train a module, we need to perform following steps:

- `bind` : Prepares environment for the computation by allocating memory.
- `init_params` : Assigns and initializes parameters.
- `init_optimizer` : Initializes optimizers. Defaults to `sgd`.
- `metric.create` : Creates evaluation metric from input metric name.
- `forward` : Forward computation.
- `update_metric` : Evaluates and accumulates evaluation metric on outputs of the last forward computation.
- `backward` : Backward computation.
- `update` : Updates parameters according to the installed optimizer and the gradients computed in the previous forward-backward batch.

This can be used as follows:

```python
# allocate memory given the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# initialize parameters by uniform random numbers
mod.init_params(initializer=mx.init.Uniform(scale=.1))
# use SGD with learning rate 0.1 to train
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# use accuracy as the metric
metric = mx.metric.create('acc')
# train 5 epochs, i.e. going over the data iter one pass
for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print('Epoch %d, Training %s' % (epoch, metric.get()))
```

To learn more about these APIs, visit [Module API](http://mxnet.io/api/python/module.html).

## High-level Interface

### Train

Module also provides high-level APIs for training, predicting and evaluating for
user convenience. Instead of doing all the steps mentioned in the above section,
one can simply call [fit API](http://mxnet.io/api/python/module.html#mxnet.module.BaseModule.fit)
and it internally executes the same steps.

To fit a module, call the `fit` function as follows:

```python
# reset train_iter to the beginning
train_iter.reset()

# create a module
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

# fit the module
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=8)
```

By default, `fit` function has `eval_metric` set to `accuracy`, `optimizer` to `sgd`
and optimizer_params to `(('learning_rate', 0.01),)`.

### Predict and Evaluate

To predict with module, we can call `predict()`. It will collect and
return all the prediction results.

```python
y = mod.predict(val_iter)
assert y.shape == (4000, 26)
```

If we do not need the prediction outputs, but just need to evaluate on a test
set, we can call the `score()` function. It runs prediction in the input validation
dataset and evaluates the performance according to the given input metric.

It can be used as follows:

```python
score = mod.score(val_iter, ['mse', 'acc'])
print("Accuracy score is %f" % (score))
```

Some of the other metrics which can be used are `top_k_acc`(top-k-accuracy),
`F1`, `RMSE`, `MSE`, `MAE`, `ce`(CrossEntropy). To learn more about the metrics,
visit [Evaluation metric](http://mxnet.io/api/python/metric.html).

One can vary number of epochs, learning_rate, optimizer parameters to change the score
and tune these parameters to get best score.

### Save and Load

We can save the module parameters after each training epoch by using a checkpoint callback.

```python
# construct a callback function to save checkpoints
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)
```

To load the saved module parameters, call the `load_checkpoint` function. It
loads the Symbol and the associated parameters. We can then set the loaded
parameters into the module.

```python
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
assert sym.tojson() == net.tojson()

# assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)
```

Or if we just want to resume training from a saved checkpoint, instead of
calling `set_params()`, we can directly call `fit()`, passing the loaded
parameters, so that `fit()` knows to start from those parameters instead of
initializing randomly from scratch. We also set the `begin_epoch` parameter so that
`fit()` knows we are resuming from a previously saved epoch.

```python
mod = mx.mod.Module(symbol=sym)
mod.fit(train_iter,
        num_epoch=8,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=3)
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
