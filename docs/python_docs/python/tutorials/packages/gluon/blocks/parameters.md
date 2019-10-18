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

# Parameter Management

<!-- adapted from diveintodeeplearning -->

The ultimate goal of training deep neural networks is finding good parameter values for a given architecture. The [nn.Sequential](/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Sequential) class is a perfect tool to work with standard models. However, very few models are entirely standard, and most scientists want to build novel things, which requires working with model parameters.

This section shows how to manipulate parameters. In particular we will cover the following aspects:

* How to access parameters in order to debug, diagnose, visualize or save them. It is the first step to understand how to work with custom models.
* We will learn how to set parameters to specific values, e.g. how to initialize them. We will discuss the structure of parameter initializers.
* We will show how this knowledge can be used to build networks that share some parameters.

As always, we start with a Multilayer Perceptron with a single hidden layer. We will use it to demonstrate the aspects mentioned above.

```{.python .input  n=1}
from mxnet import init, nd
from mxnet.gluon import nn


net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method

x = nd.random.uniform(shape=(2, 20))
net(x)            # Forward computation
```

## Parameter Access

In case of a Sequential class we can access the parameters simply by indexing each layer of the network. The `params` variable contains the required data. Let's try this out in practice by inspecting the parameters of the first layer.

```{.python .input  n=2}
print(net[0].params)
print(net[1].params)
```

From the output we can see that the layer consists of two sets of parameters: `dense0_weight` and `dense0_bias`. They are both single precision and they have the necessary shapes that we would expect from the first layer, given that the input dimension is 20 and the output dimension 256. The names of the parameters are very useful, because they allow us to identify parameters *uniquely* even in a network of hundreds of layers and with nontrivial structure. The second layer is structured in a similar way.

### Targeted Parameters

In order to do something useful with the parameters we need to access them. There are several ways to do this, ranging from simple to general. Let's look at some of them.

```{.python .input  n=3}
print(net[1].bias)
print(net[1].bias.data())
```

The first line returns the bias of the second layer. Since this is an object containing data, gradients, and additional information, we need to request the data explicitly. To request the data, we call `data` method on the parameter on the second line. Note that the bias is all 0 since we initialized the bias to contain all zeros.

We can also access the parameter by name, such as `dense0_weight`. This is possible since each layer comes with its own parameter dictionary that can be accessed directly. Both methods are entirely equivalent, but the first method leads to more readable code.

```{.python .input  n=4}
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
```

Note that the weights are nonzero as they were randomly initialized when we constructed the network.

[data](/api/python/docs/api/gluon/parameter.html#mxnet.gluon.Parameter.data) is not the only method that we can invoke. For instance, we can compute the gradient with respect to the parameters. It has the same shape as the weight. However, since we did not invoke backpropagation yet, the values are all 0.

```{.python .input  n=5}
net[0].weight.grad()
```

### All Parameters at Once

Accessing parameters as described above can be a bit tedious, in particular if we have more complex blocks, or blocks of blocks (or even blocks of blocks of blocks), since we need to walk through the entire tree in reverse order to learn how the blocks were constructed. To avoid this, blocks come with a method [collect_params](/api/python/docs/api/gluon/block.html#mxnet.gluon.Block.collect_params) which grabs all parameters of a network in one dictionary such that we can traverse it with ease. It does so by iterating over all constituents of a block and calls `collect_params` on sub-blocks as needed. To see the difference, consider the following:

```{.python .input  n=6}
# Parameters only for the first layer
print(net[0].collect_params())
# Parameters of the entire network
print(net.collect_params())
```

This provides us with the third way of accessing the parameters of the network. If we want to get the value of the bias term of the second layer we could simply use this:

```{.python .input  n=7}
net.collect_params()['dense1_bias'].data()
```

By adding a regular expression as an argument to `collect_params` method, we can select only a particular set of parameters whose names are matched by the regular expression.

```{.python .input  n=8}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

### Rube Goldberg strikes again

Let's see how the parameter naming conventions work if we nest multiple blocks inside each other. For that we first define a function that produces blocks (a block factory, so to speak) and then we combine these inside yet larger blocks.

```{.python .input  n=20}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(x)
```

Now that we are done designing the network, let's see how it is organized. `collect_params` provides us with this information, both in terms of naming and in terms of logical structure.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

We can access layers following the hierarchy in which they are structured. For instance, if we want to access the bias of the first layer of the second subblock of the first major block, we could perform the following:

```{.python .input}
rgnet[0][1][0].bias.data()
```

### Saving and loading parameters

In order to save parameters, we can use [save_parameters](/api/python/docs/api/gluon/block.html#mxnet.gluon.Block.save_parameters) method on the whole network or a particular subblock. The only parameter that is needed is the `file_name`. In a similar way, we can load parameters back from the file. We use [load_parameters](/api/python/docs/api/gluon/block.html#mxnet.gluon.Block.load_parameters) method for that:

```{.python .input}
rgnet.save_parameters('model.params')
rgnet.load_parameters('model.params')
```

## Parameter Initialization

Now that we know how to access the parameters, let's look at how to initialize them properly. By default, MXNet initializes the weight matrices uniformly by drawing from $U[-0.07, 0.07]$ and the bias parameters are all set to $0$. However, we often need to use other methods to initialize the weights. MXNet's [init](/api/python/docs/api/initializer/index.html#mxnet.initializer) module provides a variety of preset initialization methods, but if we want something unusual, we need to do a bit of extra work.

### Built-in Initialization

Let's begin with the built-in initializers. The code below initializes all parameters with Gaussian random variables.

```{.python .input  n=9}
# force_reinit ensures that the variables are initialized again,
# regardless of whether they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

If we wanted to initialize all parameters to 1, we could do this simply by changing the initializer to [Constant(1)](/api/python/docs/api/initializer/index.html#mxnet.initializer.Constant).

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

If we want to initialize only a specific parameter in a different manner, we can simply set the initializer only for the appropriate subblock (or parameter) for that matter. For instance, below we initialize the second layer to a constant value of 42 and we use the [Xavier](/api/python/docs/api/initializer/index.html#mxnet.initializer.Xavier) initializer for the weights of the first layer.

```{.python .input  n=11}
net[1].initialize(init=init.Constant(42), force_reinit=True)
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[1].weight.data()[0,0])
print(net[0].weight.data()[0])
```

### Custom Initialization

Sometimes, the initialization methods we need are not provided in the `init` module. If this is the case, we can implement a subclass of the [Initializer](/api/python/docs/api/initializer/index.html#mxnet.initializer.Initializer) class so that we can use it like any other initialization method. Usually, we only need to implement the `_init_weight` method and modify the incoming NDArray according to the initial result. In the example below, we pick a nontrivial distribution, just to prove the point. We draw the coefficients from the following distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U[-10, -5] & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

If even this functionality is insufficient, we can set parameters directly. Since `data()` returns an NDArray we can access it just like any other matrix. A note for advanced users - if you want to adjust parameters within an [autograd](/api/python/docs/api/autograd/index.html) scope you need to use [set_data](/api/python/docs/api/gluon/parameter.html#mxnet.gluon.Parameter.set_data) to avoid confusing the automatic differentiation mechanics.

```{.python .input  n=13}
net[0].weight.data()[:] += 1
net[0].weight.data()[0,0] = 42
net[0].weight.data()[0]
```

## Tied Parameters

In some cases, we want to share model parameters across multiple layers. For instance, when we want to find good word embeddings we may decide to use the same parameters both for encoding and decoding of words. In the code below, we allocate a dense layer and then use its parameters specifically to set those of another layer.

```{.python .input  n=14}
net = nn.Sequential()
# We need to give the shared layer a name such that we can reference
# its parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0,0] = 100
# And make sure that they're actually the same object rather
# than just having the same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

The above example shows that the parameters of the second and third layer are tied. They are identical rather than just being equal. That is, by changing one of the parameters the other one changes, too. What happens to the gradients is quite ingenious. Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are accumulated in the [shared.params.grad()](/api/python/docs/api/gluon/parameter.html#mxnet.gluon.Parameter.grad) during backpropagation.
