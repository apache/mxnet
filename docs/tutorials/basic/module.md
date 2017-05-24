# Module - Neural network training and inference

Training a neural network involves quite a few steps. One need to specify how
to feed input training data, initialize model parameters, perform forward and
backward passes through the network, update weights based on computed gradients, do
model checkpoints, etc. During prediction, one ends up repeating most of these
steps. All this can be quite daunting to both newcomers as well as experienced
developers.

Luckily, MXNet modularizes commonly used code for training and inference in
the `module` (`mod` for short) package. `module` provides both a
high-level and intermediate-level interfaces for executing predefined networks.

## Preliminary

In this tutorial we will demonstrate `module` usage by training a
[Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP)
on the [UCI letter recognition](https://archive.ics.uci.edu/ml/datasets/letter+recognition) dataset.

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

## High-level Interface

### Creating a Module

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

### Train, Predict, and Evaluate

Module provides high-level APIs for training, predicting and evaluating.
To fit a module, simply call the `fit` function.

```python
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=8)
```

To predict with module, simply call `predict()`. It will collect and
return all the prediction results.

```python
y = mod.predict(val_iter)
assert y.shape == (4000, 26)
```

If we do not need the prediction outputs, but just need to evaluate on a test
set, we can call the `score()` function:

```python
mod.score(val_iter, ['mse', 'acc'])
```

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

## Intermediate-level Interface

We already saw how to use module for basic training and inference. Now we are
going to see a more flexible usage of module. Instead of calling
the high-level `fit` and `predict` APIs, one can write a training program with the intermediate-level APIs
`forward` and `backward`.

```python
# create module
mod = mx.mod.Module(symbol=net)
# allocate memory by given the input data and label shapes
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

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
