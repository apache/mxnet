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

# Custom Layers

<!-- adapted from diveintodeeplearning -->

One of the reasons for the success of deep learning can be found in the wide range of re-usable layers that can be used in a deep network. This allows for a tremendous degree of customization and adaptation. Sooner or later you will encounter a layer that doesn't exist yet in Gluon or one that you want to create. This is when it's time to build a custom layer. This section shows you how.

Defining a layer is as easy as subclassing [`nn.Block`](/api/gluon/mxnet.gluon.nn.Block.html#mxnet.gluon.nn.Block) or [`nn.HybridBlock`](/api/gluon/mxnet.gluon.nn.HybridBlock.html#mxnet.gluon.nn.HybridBlock) and implementing `forward` or `hybrid_forward`, respectively. To take advantage of the performance gains with `nn.HybridBlock` see the section on [Hybridization](hybridize.html). 

Note that we've gone through rationale for defining layers, but `nn.Block`'s work even for non-sequential network. In fact, you can use a `Block` to encapsualte any re-usable architecture you want.

We will discuss making custom layers using `nn.Block` below. 

## Layers without Parameters

Since this is slightly intricate, we start with a custom layer that doesn't have any inherent parameters. Our first step is very similar to when we [introduced blocks](nn.md) previously. The following `CenteredLayer` class constructs a layer that subtracts the mean from the input. We build it by inheriting from the `Block` class and overriding the `forward`  and `__init__` methods.

```{.python .input  n=1}
from mxnet import gluon, nd
from mxnet.gluon import nn


class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

To see how it works let's feed some data into the layer.

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))
print(layer)
```

We can also use it to construct more complex models.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128), 
        CenteredLayer())
net.initialize()
print(net)
```

Let's see whether the centering layer did its job. For that we send random data through the network and check whether the mean is $0$. Note that since we're dealing with floating point numbers, we're going to see a very small albeit typically nonzero number.

```{.python .input  n=4}
y = net(nd.random.uniform(shape=(4, 8)))
y.mean().asscalar()
```

## Layers with Parameters

Now that we know how to define layers in principle, let's define layers with parameters. These can be adjusted through training. In order to simplify things for an avid deep learning researcher, the [`Parameter`](/api/gluon/mxnet.gluon.parameter.html) class and the `ParameterDict` dictionary provide some basic housekeeping functionality. In particular, they govern access, initialization, sharing, saving and loading model parameters. For instance, this way we don't need to write custom serialization routines for each new custom layer.

We can access the parameters via the `params` variable of the `ParameterDict` in `Block`. The parameter dictionary is just that - a dictionary that maps string type parameter names to model parameters in the `Parameter` type.  We can create a `Parameter` instance from `ParameterDict` via the `get` function which attempts to retrieve a parameter, or create it if not found.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
print(params)
```

Let's use this to implement our own version of the dense layer. It has two parameters - bias and weight. To make it a bit nonstandard, we bake in the ReLU activation as default. Next, we implement a fully connected layer with both weight and bias parameters.  It uses ReLU as an activation function, where `in_units` and `units` are the number of inputs and the number of outputs, respectively.

```{.python .input  n=19}
class MyDense(nn.Block):

    def __init__(self, units, in_units, **kwargs):
        # units: the number of outputs in this layer
        # in_units: the number of inputs in this layer

        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

Naming the parameters allows us to access them by name through dictionary lookup later. It's a good idea to give them instructive names. Next, we instantiate the `MyDense` class and access its model parameters.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

We can directly carry out forward calculations using custom layers.

```{.python .input  n=20}
dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))
print(dense)
```

We can also construct models using custom layers. Once we have that we can use it just like the built-in dense layer. The only exception is that in our case, shape inference is not automatic as we have explicitly defined the shape of the weight matrix during initialization.

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
print(net)
```

