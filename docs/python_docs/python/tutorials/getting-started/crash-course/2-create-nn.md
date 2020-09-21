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

In this step, you learn how to use NP on Apache MXNet to create neural networks in
Gluon. In addition to the `np` package that you learned about in the previous
step [Step 1: Manipulate data with NP on MXNet](1-ndarray.md), you also need to import
the neural networks modules from `gluon`. Gluon contains large number of built-in neural
network layers in the following two modules:

1. `mxnet.gluon.nn`
2. `mxnet.gluon.contrib.nn`

Use the following commands to import the packages required for this step.

```python
from mxnet import np, npx
from mxnet.gluon import nn, Block
npx.set_np()  # Change MXNet to the numpy-like mode.
```

## Create your neural network's first layer

Before you create your first neural network, you have to first create a layer
inside the neural network. One of the simplest layers you can create is a
**Dense** layer or **densely-connected** layer. A dense layer consists of a layer where every node
in the input is connected to every node in the following layer or output. Use the
following code example to start building a dense layer with two output units.

```python
layer = nn.Dense(5)
layer
```

In the example above, the **-1** denotes that the size of the input layer is not
specified during initialization.

You can also call the **Dense** layer with an `in_units` parameter if you know
the shape of your input unit.

```python
layer = nn.Dense(5,in_units=3)
layer
```

In addition to the `in_units` param, you can also add an activation function to
the layer using the `activation` param. The Dense layer implements the operation

$$ output = \sigma(W \cdot X + b) $$

Call the Dense layer wtih an `activation` parameter to use an activation
function.

```python
layer = nn.Dense(5, in_units=3,activation='relu')
```

Voila! Congratulations on creating a simple neural network with Gluon. But for most of your use
cases, you will need to create a neural network with more than one dense layer
or with multiple types of other layers. In addition to the `Dense` layer, you can find more layers
available from Gluon at [mxnet nn layers TODO: Change 1.6
link](https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html#module-
mxnet.gluon.nn)

So now that you have created a neural network, you are probably wondering how to pass data into your network?

First, you need to initialize the network weights, if you use the default
initialization method which draws random values uniformly in the range $[-0.7, 0.7]$.
You can see this in the following example.

**Note**: You will dive deeper into initialization in the next notebook

```python
layer.initialize()
```

Now that you have initialized your network you can give it data. Data passing through a network
is also called a forward pass. You can do a forward pass with random data, shown in the
following example. First, you create a $(10,3)$ shape random input `x` and feed it into the
layer to compute the output.

```python
x = np.random.uniform(-1,1,(10,3))
layer(x)
```

The layer produced a $(10,5)$ shape output from your $(10,3)$ input.

**When you dont specify the `in_unit` parameter, the system  automatically
infers it during the first time you feed in data during the first forward step
after you create and initialize the weights.**

You can access the weight after the first forward pass, as shown in this
example.

```python
layer.weight.data()
```

## Chain layers into a neural network using nn.Sequential

Sequential provides a special way of rapidly building networks when the network 
architecture follows a common design pattern: the layers look like a stack of 
pancakes. Many networks follow this pattern: a bunch of layers, one stacked on 
top of another, where the output of each layer is fed directly to the input of 
the following layer. To use sequential, simply provide a list of layers (we pass them in by calling 
`net.add(<Layer goes here!>`). To do this you can use your previous example of Dense layers 
and create a 3-layer multi layer perceptron. You can create a sequential block 
using `nn.Sequential()` method and add layers using `add()` method.

```python
net = nn.Sequential()

net.add(nn.Dense(5,in_units=3,activation='relu'),
        nn.Dense(25, activation='relu'),
        nn.Dense(2)
       )

net
```

The layers are ordered exactly the way you defined your neural network with
numbers starting from 0. You can access the layers by indexing the network using
`[]`.

```python
net[1]
```

## Create a custom neural network architecture flexibly

`nn.Sequential()` allows you to create your multi-layer neural network with
existing layers from `gluon.nn`. It also includes a pre-defined `forward()`
function that sequentially execututes thse layers. But what if you want to add
more computation during your forward pass or create a new/novel network. How do
you create a network?

In gluon, every neural network layer is defined by using a base class
`nn.Block()`. In gluon, a Block has one main job - define a forward method that
takes some NDArray input `x` and generates an NDArray output. A Block can do
something simple like apply an activation function. Tt can also combine a
bunch of other Blocks together in creative ways. In this case, you will
construct three Dense layers. The `forward()` method can invoke these layers in
turn to generate the output.

Create a subclass of `nn.Block` and implement two methods by using the following
code.

- `__init__` create the layers
- `forward` define the forward function.

```
class Net(gluon.Block):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

```python
class MLP(Block):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(5,activation='relu')
        self.dense2 = nn.Dense(25,activation='relu')
        self.dense3 = nn.Dense(2)

    def forward(self, x):
        layer1 = self.dense1(x)
        layer2 = self.dense2(layer1)
        layer3 = self.dense3(layer2)
        return x
    
net = MLP()
net
```

You can the following code to implement a famous network called
[LeNet](http://yann.lecun.com/exdb/lenet/) through `nn.Sequential`.

```python
class LeNet(Block):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2D(channels=6, kernel_size=3, activation='relu')
        self.pool1  = nn.MaxPool2D(pool_size=2, strides=2)
        self.conv2  = nn.Conv2D(channels=16, kernel_size=3, activation='relu')
        self.pool2  = nn.MaxPool2D(pool_size=2, strides=2)
        self.dense1 = nn.Dense(120, activation="relu")
        self.dense2 = nn.Dense(84, activation="relu")
        self.dense3 = nn.Dense(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
Lenet = LeNet()
```

```python
image_data = np.random.uniform(-1,1, (1,1,28,28))

Lenet.initialize()

Lenet(image_data)
```

In this example, you can use `print` to get the intermediate results between
layers. This approach provides a more flexible way to define the forward
function.

You can use `[]` to index a particular layer. For example, the following
accesses the first layer's weight and sixth layer's bias.

```python
Lenet.conv1.weight.data().shape, Lenet.dense1.bias.data().shape

```

## Using predefined (pretrained) architectures

Till now, you have seen how to create custom neural network architectures. But
what if you want to replicate or baseline your dataset using some of the common
models found in computer vision or natural language processing (nlp). Gluon includes
common architectures that you can directly use. The Gluon Model Zoo provides a
collection of off-the-shelf models e.g. RESNET, BERT, etc. Some of the common
architectures include:

- [Gluon CV model zoo](https://gluon-cv.mxnet.io/model_zoo/index.html)

- [Gluon NLP model zoo](https://gluon-nlp.mxnet.io/model_zoo/index.html)

Look at the following example

```python
from mxnet.gluon import model_zoo

net = model_zoo.vision.resnet50_v2(pretrained=True)
net.hybridize()

dummy_input = np.ones(shape=(1,3,224,224))
output = net(dummy_input)
output.shape
```

## Deciding the paradigm for your NN

In MXNet you can use the Gluon API (Imperative programming) that is very user
friendly for people familiar with python programming. The Gluon API allows for quick 
prototyping, easy debugging and has a natural control flow. Additionally, in the backend, MXNet
can convert the network from using Symbolic or Declarative programming into
static graphs with low level optimizations on operators. However, static graphs
are less flexible because any logic must be encoded into the graph as special
operators like scan, while_loop, and cond. It’s also hard to debug.

So how can you make use of Symbolic programming while getting the flexibility of
Imperative programming to quickly protype and debug?

Enter **HybridBlocks**

HybridBlocks can run in a fully imperatively way. Where you define their computation with real
functions acting on real inputs. But they’re also capable of running
symbolically, acting on placeholders. Gluon hides most of this under the hood so
you’ll only need to know how it works when you want to write your own layers.

```python
net_hybrid_seq = nn.HybridSequential()

net_hybrid_seq.add(nn.Dense(5,in_units=3,activation='relu'),
        nn.Dense(25, activation='relu'),
        nn.Dense(2)
       )

net_hybrid_seq
```

To compile and optimize `HybridSequential`, you can then call its `hybridize
method`.

```python
net_hybrid_seq.hybridize()
```

Performance

To get a sense of the speedup from hybridizing, you can compare the performance
before and after hybridizing by measuring the time it takes to
make 1000 forward passes through the network.

```python
from time import time

def benchmark(net, x):
    y = net(x)
    start = time()
    for i in range(1,1000):
        y = net(x)
    return time() - start

x_bench = np.random.normal(size=(1,512))

net_hybrid_seq = nn.HybridSequential()

net_hybrid_seq.add(nn.Dense(256,activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(2)
       )
net_hybrid_seq.initialize()

print('Before hybridizing: %.4f sec'%(benchmark(net_hybrid_seq, x_bench)))
net_hybrid_seq.hybridize()
print('After hybridizing: %.4f sec'%(benchmark(net_hybrid_seq, x_bench)))
```

Peeling back another layer, you also have a `HybridBlock` which is the hybrid version
of the `Block` API.

With normal Blocks, you need to define a `forward` function that takes an
input `x` and computes the result of the forward pass through the network. To
define a `HybridBlock`, you create the same `forward` function. MXNet takes care
of hybridizing the model at the backend so you don't have to make changes to
your code to convert it to a symbolic paradigm

```python
from mxnet.gluon import HybridBlock

class MLP_Hybrid(HybridBlock):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(256,activation='relu')
        self.dense2 = nn.Dense(128,activation='relu')
        self.dense3 = nn.Dense(2)

    def forward(self, x):
        layer1 = self.dense1(x)
        layer2 = self.dense2(layer1)
        layer3 = self.dense3(layer2)
        return layer3
    
net_Hybrid = MLP_Hybrid()
net_Hybrid.initialize()

print('Before hybridizing: %.4f sec'%(benchmark(net_Hybrid, x_bench)))
net_Hybrid.hybridize()
print('After hybridizing: %.4f sec'%(benchmark(net_Hybrid, x_bench)))
```

Given a HybridBlock whose forward computation consists of going through other
HybridBlocks, you can compile that section of the network by calling the
HybridBlocks `.hybridize()` method.

All of MXNet’s predefined layers are HybridBlocks. This means that any network
consisting entirely of predefined MXNet layers can be compiled and run at 
faster speeds by calling `.hybridize()`.

## Saving and Loading your models

Now that you've trained your model, it is a good idea to save the trained model
so that you can host the model for inference or so that you can avoid training the model again.
Another reason would be to train your model using one language (like Python that
has a lot of tools for training) and run inference using a different language.

There are two ways to save your model in MXNet.
1. Save/load the model weights/parameters only
2. Save/load the model weights/parameters and the architectures

#### 1. Save/load the model weights/parameters only

You can use the `save_parameters` and `load_parameters` methods to save and load the
model weights. Take your simplest model `layer` and save your parameters
first. The model parameters are the params that you save **after** you train
your model.

```python
file_name = 'layer.params'
layer.save_parameters(file_name)
```

And now load this model again. To load the parameters into a model, you
will first have to build the model. To do this you will need to create a
simple function to build it.

```python
def build_model():
    layer = nn.Dense(5, in_units=3,activation='relu')
    return layer

layer_new = build_model()
```

```python
layer_new.load_parameters('layer.params')
```

**Note**: The `save_parameters` and `load_parameters` methods are used for models
that use a `Block` method instead of  `HybridBlock` method to build the model.
These models may have complex architectures where the model architectures could 
change during execution. E.g. if you have a model that uses an if-else
conditional statement to choose between two different architectures.

#### 2. Save/load the model weights/parameters and the architectures

For models that use the **HybridBlock**, the model architecture stays static and
does not change during execution. Therefore, both model parameters AND architectures
can be saved and loaded using the `export` and `imports` methods.

Now look at your `MLP_Hybrid` model and export the model using the
export function. The export function will export the model architecture into a
.json file and the model parameters into a .params file.

```python
net_Hybrid.export('MLP_hybrid')
```

```python
net_Hybrid.export('MLP_hybrid')
```

Similarly, to load this model back, you can use `gluon.nn.SymbolBlock`. To
demonstrate that, load the network you serialized above.

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    net_loaded = nn.SymbolBlock.imports("MLP_hybrid-symbol.json", ['data'], "MLP_hybrid-0000.params",ctx=None)
```

```python
net_loaded(x_bench)
```

## Visualizing your models

In MXNet, you can use the `Block.summary()` method to print a summary of the model's
outputs and parameters. Currently, the method works for networks that use
`HybridBlocks` and `Blocks`; however, the method should be called **before** the
`net.hybridize()` method and after the network has been initialized. The
`Block.summary()` method requires one forward pass through your network. You
will be using your original data from above.

Look at the following examples

- layer: your single layer network
- Lenet: a non-hybridized LeNet network
- net_Hybrid: your MLP Hybrid network

```python
layer.summary(x)
```

```python
Lenet.summary(image_data)
```

You you are able to print the summaries of the two networks `layer` and `Lenet`
easily since you didn't hybridize them above. However, the last network
`net_Hybrid` was hybridized above and throws an `AssertionError` if you
try `net_Hybrid.summary(x_bench)`. Now you can call another instance of the same
network and instantiate it for your summary and then hybridize it.

```python
net_Hybrid_summary = MLP_Hybrid()

net_Hybrid_summary.initialize()

net_Hybrid_summary.summary(x_bench)

net_Hybrid_summary.hybridize()
```

## Next steps: TODO: UPDATE

Now that you have created a neural network, learn how to automatically
compute the gradients in [Step 3: Automatic differentiation with
autograd](3-autograd.md).

```python

```
