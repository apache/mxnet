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

# Step 2: Create a neural network

In this step, you learn how to use NP on MXNet to create neural networks in Gluon. In addition to the `np` package that you learned about in the previous step [Step 1: Manipulate data with NP on MXNet](1-ndarray.md), you also import the neural network `nn` package from `gluon`.

Use the following commands to import the packages required for this step.

```{.python .input  n=2}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()  # Change MXNet to the numpy-like mode.
```

## Create your neural network's first layer

Use the following code example to start with a dense layer with two output units.
<!-- mention what the none and the linear parts mean? -->

```{.python .input  n=31}
layer = nn.Dense(2)
layer
```

Initialize its weights with the default initialization method, which draws random values uniformly from $[-0.7, 0.7]$. You can see this in the following example.

```{.python .input  n=32}
layer.initialize()
```

Do a forward pass with random data, shown in the following example. We create a $(3,4)$ shape random input `x` and feed into the layer to compute the output.

```{.python .input  n=34}
x = np.random.uniform(-1,1,(3,4))
layer(x)
```

As can be seen, the layer's input limit of two produced a $(3,2)$ shape output from our $(3,4)$ input. You didn't specify the input size of `layer` before, though you can specify it with the argument `in_units=4` here. The system  automatically infers it during the first time you feed in data, create, and initialize the weights. You can access the weight after the first forward pass, as shown in this example.

```{.python .input  n=35}
# layer.weight.data() # FIXME
```

## Chain layers into a neural network

Consider a simple case where a neural network is a chain of layers. During the forward pass, you run layers sequentially one-by-one. Use the following code to implement a famous network called [LeNet](http://yann.lecun.com/exdb/lenet/) through `nn.Sequential`.

```{.python .input}
net = nn.Sequential()
# Add a sequence of layers.
net.add(# Similar to Dense, it is not necessary to specify the input channels
        # by the argument `in_channels`, which will be  automatically inferred
        # in the first forward pass. Also, we apply a relu activation on the
        # output. In addition, we can use a tuple to specify a  non-square
        # kernel size, such as `kernel_size=(2,4)`
        nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        # One can also use a tuple to specify non-symmetric pool and stride sizes
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # The dense layer will automatically reshape the 4-D output of last
        # max pooling layer into the 2-D shape: (x.shape[0], x.size/x.shape[0])
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10))
net
```

<!--Mention the tuple option for kernel and stride as an exercise for the reader? Or leave it out as too much info for now?-->

Using `nn.Sequential` is similar to `nn.Dense`. In fact, both of them are subclasses of `nn.Block`. Use the following code to initialize the weights and run the forward pass.

```{.python .input}
net.initialize()
# Input shape is (batch_size, color_channels, height, width)
x = np.random.uniform(size=(4,1,28,28))
y = net(x)
y.shape
```

You can use `[]` to index a particular layer. For example, the following
accesses the first layer's weight and sixth layer's bias.

```{.python .input}
(net[0].weight.data().shape, net[5].bias.data().shape)
```

## Create a neural network flexibly

In `nn.Sequential`, MXNet will automatically construct the forward function that sequentially executes added layers.
Here is another way to construct a network with a flexible forward function.

Create a subclass of `nn.Block` and implement two methods by using the following code.

- `__init__` create the layers
- `forward` define the forward function.

```{.python .input  n=6}
class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        # Run `nn.Block`'s init method
        super(MixMLP, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(nn.Dense(3, activation='relu'),
                     nn.Dense(4, activation='relu'))
        self.dense = nn.Dense(5)
    def forward(self, x):
        y = npx.relu(self.blk(x))
        print(y)
        return self.dense(y)

net = MixMLP()
net
```

In the sequential chaining approach, you can only add instances with `nn.Block` as the base class and then run them in a forward pass. In this example, you used `print` to get the intermediate results and `nd.relu` to apply relu activation. This approach provides a more flexible way to define the forward function.

The following code example uses `net` in a similar manner as earlier.

```{.python .input}
net.initialize()
x = np.random.uniform(size=(2,2))
net(x)
```

Finally, access a particular layer's weight with this code.

```{.python .input  n=8}
net.blk[1].weight.data()
```

## Next steps

After you create a neural network, learn how to automatically
compute the gradients in [Step 3: Automatic differentiation with autograd](3-autograd.md).
