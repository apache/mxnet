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

# Trainer

Training a neural network model consists of iteratively performing three simple steps.

The first step is the forward step which computes the loss.  In MXNet Gluon, this first step is achieved by doing a forward pass by calling `net.forward(X)` or simply `net(X)` and then calling the loss function with the result of the forward pass and the labels. For example `l = loss_fn(net(X), y)`.

The second step is the backward step which computes the gradient of the loss with respect to the parameters. In Gluon, this step is  achieved by doing the first step in an [autograd.record()](/api/python/docs/api/autograd/index.html) scope to record the computations needed to calculate the loss, and then calling `l.backward()` to compute the gradient of the loss with respect to the parameters.

The final step is to update the neural network model parameters using an optimization algorithm. In Gluon, this step is performed by the [gluon.Trainer](/api/python/docs/api/gluon/trainer.html) and is the subject of this guide. When creating a  Gluon `Trainer` you must provide a collection of parameters that need to be learnt. You also provide an `Optimizer` that will be used to update the parameters every training iteration when `trainer.step` is called.

## Basic Usage

### Network and Trainer

To illustrate how to use the Gluon `Trainer` we will create a simple perceptron model and create a `Trainer ` instance using the perceptron model parameters and a simple optimizer - `sgd` with learning rate as 1.

```{.python .input}
from mxnet import nd, autograd, optimizer, gluon

net = gluon.nn.Dense(1)
net.initialize()

trainer = gluon.Trainer(net.collect_params(),
                        optimizer='sgd', optimizer_params={'learning_rate':1})

```

### Forward and Backward Pass

Before we can use the `trainer` to update model parameters, we must first run the forward and backward passes. Here we implement a function to compute the first two steps (forward step and backward step) of training the perceptron on a random dataset.

```{.python .input}
batch_size = 8
X = nd.random.uniform(shape=(batch_size, 4))
y = nd.random.uniform(shape=(batch_size,))

loss = gluon.loss.L2Loss()

def forward_backward():
    with autograd.record():
        l = loss(net(X), y)
    l.backward()

forward_backward()
```

**Warning**: It is extremely important that the gradients of the loss function with respect to your model parameters are computed before running `trainer step`. A common way to introduce bugs to your model training code is to omit the `loss.backward()`before the update step.



Before updating, let's check the current network parameters.

```{.python .input}
curr_weight = net.weight.data().copy()
print(curr_weight)
```

### `Trainer` step

Now we will call the `step` method to perform one update. We provide the `batch_size` as an argument to normalize the size of the gradients and make it independent of the batch size. Otherwise we'd get larger gradients with larger batch sizes. We can see the network parameters have now changed.

```{.python .input}
trainer.step(batch_size)
print(net.weight.data())
```

Since we used plain SGD, the update rule is $w = w - \eta/b \nabla \ell$, where $b$ is the batch size and $\nabla\ell$ is the gradient of the loss function with respect to the weights and $\eta$ is the learning rate.

We can verify it by running the following code snippet which is explicitly performing the SGD update.

```{.python .input}
print(curr_weight - net.weight.grad() * 1 / batch_size)
```



## Advanced Usage

### Using Optimizer Instance

In the previous example, we use the string argument `sgd` to select the optimization method, and `optimizer_params` to specify the optimization method arguments.

All pre-defined optimization methods can be passed in this way and the complete list of implemented optimizers is provided in the [mxnet.optimizer](/api/python/docs/api/optimizer/index.html) module.

However we can also pass an optimizer instance directly to the `Trainer` constructor.

For example:

```{.python .input}
optim = optimizer.Adam(learning_rate = 1)
trainer = gluon.Trainer(net.collect_params(), optim)
```

```{.python .input}
forward_backward()
trainer.step(batch_size)
net.weight.data()
```

For reference and implementation details about each optimizer, please refer to the [guide](/api/python/docs/api/optimizer/index.html) for the `optimizer` module.

### KVStore Options

The `Trainer` constructor also accepts the following keyword arguments for :

- `kvstore` – how key value store  should be created for multi-gpu and distributed training. Check out  [mxnet.kvstore.KVStore](/api/python/docs/api/kvstore/index.html) for more information. String options are any of the following ['local', 'device', 'dist_device_sync', 'dist_device_async'].
- `compression_params` – Specifies type of gradient compression and additional arguments depending on the type of compression being used. See [mxnet.KVStore.set_gradient_compression_method](/api/python/docs/api/kvstore/index.html#mxnet.kvstore.KVStore.set_gradient_compression) for more details on gradient compression.
- `update_on_kvstore` – Whether to perform parameter updates on KVStore. If None, then the `Trainer` instance  will choose the more suitable option depending on the type of KVStore.

### Changing the Learning Rate

We set the initial learning rate when creating a trainer by passing the learning rate as an `optimizer_param`. However, sometimes we may need to change the learning rate during training, for example when doing an explicit learning rate warmup schedule.  The trainer instance provides an easy way to achieve this.

The current training rate can be accessed through the `learning_rate` attribute.

```{.python .input}
trainer.learning_rate
```

We can change it through the `set_learning_rate` method.

```{.python .input}
trainer.set_learning_rate(0.1)
trainer.learning_rate
```



In addition, there are multiple pre-defined learning rate scheduling methods that are already implemented in the [mxnet.lr_scheduler](/api/python/docs/api/lr_scheduler/index.html) module. The learning rate schedulers can be incorporated into your trainer by passing them in as an `optimizer_param` entry. Please refer to the [LR scheduler guide](/api/python/docs/tutorials/packages/gluon/training/learning_rates/learning_rate_schedules.html) to learn more.



## Summary

* The MXNet Gluon `Trainer` API is used to update the parameters of a network with a particular optimization algorithm.
* After the forward and backward pass, the model update step is done in Gluon using `trainer.step()`.
* A Gluon `Trainer` can be instantiated by passing in the name of the optimizer to use and the `optimizer_params` for that optimizer or alternatively by passing in an instance of `mxnet.optimizer.Optimizer`.
* You can change the learning rate for a Gluon `Trainer` by setting the member variable but Gluon also provides a module for learning rate scheduling.



## Next Steps

While optimization and optimizers play a significant role in deep learning model training, there are still other important components to model training. Here are a few suggestions about where to look next.

* The [Optimizer API](/api/python/docs/api/optimizer/index.html) and [optimizer guide](/api/python/docs/tutorials/packages/optimizer/index.html) have information about all the different optimizers implemented in MXNet and their update steps. The [Dive into Deep Learning](https://en.diveintodeeplearning.org/chapter_optimization/index.html) book also has a chapter dedicated to optimization methods and explains various key optimizers in great detail.

- Take a look at the [guide to parameter initialization](/api/python/docs/tutorials/packages/gluon/blocks/init.html) in MXNet to learn about what initialization schemes are already implemented, and how to implement your custom initialization schemes.
- Also check out this  [guide on parameter management](/api/python/docs/tutorials/packages/gluon/blocks/parameters.html) to learn about how to manage model parameters in gluon.
- Make sure to take a look at the [guide to scheduling learning rates](/api/python/docs/tutorials/packages/gluon/training/learning_rates/learning_rate_schedules.html) to learn how to create learning rate schedules to make your training converge faster.
- Finally take a look at the [KVStore API](/api/python/docs/api/kvstore/index.html) to learn how parameter values are synchronized over multiple devices.
