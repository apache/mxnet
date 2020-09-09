# Step 2: Create a neural network

In this step, you learn how to use NP on MXNet to create neural networks in
Gluon. In addition to the `np` package that you learned about in the previous
step [Step 1: Manipulate data with NP on MXNet](1-ndarray.md), you also import
the neural networks from `gluon`. Gluon contains large number of build-in neural
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

Before we create our first neural network, we have to first create a layer
inside the neural network. One of the simplest layers you can create is a
**Dense** layer or **densely-connected** layer. A dense layer consists of nodes
in the input that are connected to every node in the next layer. Use the
following code example to start with a dense layer with two output units.

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

Voila! Congratulations on creating a simple neural network. But for your use
cases, you will need to create a neural network with more than one dense layer
and other layers. In additional to the `Dense` layer, you can find more layers
at [mxnet nn layers TODO: Change 1.6
link](https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html#module-
mxnet.gluon.nn)

So now that i have created a neural network, can I start passing data?

Almost!! we just need to initialize the network weights with the default
initialization method, which draws random values uniformly from $[-0.7, 0.7]$.
You can see this in the following example.

**Note**: We will dive a little deeper into initialization in the next notebook

```python
layer.initialize()
```

Now you can pass your data through a network. Passing a data through a network
is also called a forward pass. Do a forward pass with random data, shown in the
following example. We create a $(10,3)$ shape random input `x` and feed into the
layer to compute the output.

```python
x = np.random.uniform(-1,1,(10,3))
layer(x)
```

The layer produced a $(10,5)$ shape output from our $(10,3)$ input.

**When you dont specify the `in_unit` parameter, the system  automatically
infers it during the first time you feed in data during the first forward step
after you create and initialize the weights.**

You can access the weight after the first forward pass, as shown in this
example.

```python
layer.weight.data()
```

## Chain layers into a neural network using nn.Sequential

Consider a simple case where a neural network is a chain of layers. Sequential
gives us a special way of rapidly building networks when follow a common design
pattern: they look like a stack of pancakes. Many networks follow this pattern:
a bunch of layers, one stacked on top of another, where the output of each layer
is the input to the next layer. Sequential just takes a list of layers (we pass
them in by calling `net.add(<Layer goes here!>`). Let's take our previous
example of Dense layers and create a 3-layer multi layer perceptron. You can
create a sequential block using `nn.Sequential()` method and add layers using
`add()` method.

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
function that sequentially execututes added layers. But what if we want to add
more computation during your forward pass or create a new/novel network. How do
we create a network?

In gluon, every neural network layer is defined by using a base class
`nn.Block()`. In gluon a Block has one main job - define a forward method that
takes some NDArray input x and generates an NDArray output. A Block can just do
something simple like apply an activation function. But it can also combine a
bunch of other Blocks together in creative ways. In this case, we’ll just want
to instantiate three Dense layers. The forward can then invoke the layers in
turn to generate its output.

The basic structure that you can use for any neural network is the following:

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

Simirly, you can the following code to implement a famous network called
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

Till now, we have seen how to create your own neural network architectures. But
what if we want to replicate or baseline your dataset using some of the common
models in computer visions or natural language processing (nlp). Gluon includes
common architectures that you can directly use. The Gluon Model Zoo provides a
collection of off-the-shelf models e.g. RESNET, BERT etc. Some of the common
architectures include:

- [Gluon CV model zoo](https://gluon-cv.mxnet.io/model_zoo/index.html)

- [Gluon NLP model zoo](https://gluon-nlp.mxnet.io/model_zoo/index.html)

Lets look at an example

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
friendly and allows for quick prototyping, easy debugging and natural control
flow for people familiar with python programming. However, at the backend, MXNET
can also convert the network using Symbolic or Declarative programming into
static graphs with low level optimizations on operators. However, static graphs
are less flexible because any logic must be encoded into the graph as special
operators like scan, while_loop and cond. It’s also hard to debug.

So how can we make use of Symbolic programming while getting the flexibility of
an Imperative programming to quickly protype and debug?

Enter **HybridBlocks**

Each HybridBlock can run fully imperatively defining their computation with real
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

To compile and optimize the `HybridSequential`, we can then call its `hybridize
method`.

```python
net_hybrid_seq.hybridize()
```

Performance

To get a sense of the speedup from hybridizing, we can compare the performance
before and after hybridizing by measuring in either case the time it takes to
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

Peeling another layer, we also have a `HybridBlock` which is the hybrid version
of the `Block` API.

With normal Blocks, we just need to define a `forward` function that takes an
input `x` and computes the result of the forward pass through the network. To
define a `HybridBlock`, we create the same `forward` function. MXNet takes care
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
consisting entirely of predefined MXNet layers can be compiled and run at much
faster speeds by calling `.hybridize()`.

## Saving and Loading your models

Now that you've trained your model, it is a good idea to save the trained model
so that you can host the model on cloud or avoid training the model again from
scratch if you want to retrain the model that is suffering from concept drift.
Another reason would be to train your model using one language (like Python that
has a lot of tools for training) and run inference using a different language
(like Scala probably because your application is built on Scala).

There are two ways to save your model in mxnet.
1. Save/load the model weights/parameters only
2. Save/load the model weights/parameters and the architectures

#### 1. Save/load the model weights/parameters only

You can use `save_parameters` and `load_parameters` method to save and load the
model weights. Let's take our simplest model `layer` and save our parameters
first. The model parameters are the params that you save **after** you train
your model.

```python
file_name = 'layer.params'
layer.save_parameters(file_name)
```

And now let's load this model again. To load the parameters into a model, you
will first have to build the model. Since we have a simple model, let's create a
simple function to build it

```python
def build_model():
    layer = nn.Dense(5, in_units=3,activation='relu')
    return layer

layer_new = build_model()
```

```python
layer_new.load_parameters('layer.params')
```

**Note**: The `save_parameters` and `load_parameters` method is used for models
that use a `Block` method instead of  `HybridBlock` method to build the model.
These models may have complex architectures where the model architectures may
change during execution. E.g. if you have a model that uses an if-else
conditional statement to choose between two different architectures.

#### 2. Save/load the model weights/parameters and the architectures

For models that use the **HybridBlock**, the model architecture stays static and
do no change during execution. Therefore both model parameters AND architecture
can be saved and loaded using `export`, `imports` methods.

Let's look at our `MLP_Hybrid` model and export the model into files using the
export function. The export function will export the model architecture into a
.json file and model parameters into a .params file.

```python
net_Hybrid.export('MLP_hybrid')
```

```python
net_Hybrid.export('MLP_hybrid')
```

Similarly, to load this model back, you will use `gluon.nn.SymbolBlock`. To
demonstrate that, let’s load the network we serialized above.

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

In MXNet, you can use `Block.summary()` method to print a summary of the model's
outputs and parameters. Currently, the method works for networks that use
`HybridBlocks` and `Blocks`, however, the method should be called **before** the
`net.hybridize()` method and after the network has been initialized. The
`Block.summary()` method requires one forward pass through your network so we
will be providing our original data that we have used above.

Let's look at the following examples

- layer: our single layer network
- Lenet: a non-hybridized LeNet network
- net_Hybrid: our MLP Hybrid network

```python
layer.summary(x)
```

```python
Lenet.summary(image_data)
```

We were able to print the summaries of the two networks `layer` and `Lenet`
easily since we didn't hybridize them above. However the last network
`net_Hybrid` was hybridized above and will throw out an `AssertionError` if you
try `net_Hybrid.summary(x_bench)`. Let's first call another instance of the same
network and instantiate it for our summary and then hybridize it

```python
net_Hybrid_summary = MLP_Hybrid()

net_Hybrid_summary.initialize()

net_Hybrid_summary.summary(x_bench)

net_Hybrid_summary.hybridize()
```

## Next steps: TODO: UPDATE

After you create a neural network, learn how to automatically
compute the gradients in [Step 3: Automatic differentiation with
autograd](3-autograd.md).

```python

```
