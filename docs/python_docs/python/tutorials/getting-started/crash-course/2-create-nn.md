# Step 2: Create a neural network

In this step, you learn how to use NP on MXNet to create neural networks in
Gluon. In addition to the `np` package that you learned about in the previous
step [Step 1: Manipulate data with NP on MXNet](1-ndarray.md), you also import
the neural networks from `gluon`. Gluon contains large number of build-in neural
network layers in the following two modules:

1. `mxnet.gluon.nn`
2. `mxnet.gluon.contrib.nn`

Use the following commands to import the packages required for this step.

```{.python .input  n=1}
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

```{.python .input  n=2}
layer = nn.Dense(5)
layer
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "Dense(-1 -> 5, linear)"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In the example above, the **-1** denotes that the size of the input layer is not
specified during initialization.

You can also call the **Dense** layer with an `in_units` parameter if you know
the shape of your input unit.

```{.python .input  n=3}
layer = nn.Dense(5,in_units=3)
layer
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "Dense(3 -> 5, linear)"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In addition to the `in_units` param, you can also add an activation function to
the layer using the `activation` param. The Dense layer implements the operation

$$ output = \sigma(W \cdot X + b) $$

Call the Dense layer wtih an `activation` parameter to use an activation
function.

```{.python .input  n=4}
layer = nn.Dense(5, in_units=3,activation='relu')
```

Viola! Congratulations on creating a simple neural network. But for your use
cases, you will need to create a neural network with more than one dense layer
and other layers. In additional to the `Dense` layer, you can find more layers
at [mxnet nn layers TODO: Change 1.6
link](https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html#module-
mxnet.gluon.nn)

So now that i have created a neural network, can I start passing data?

Almost!! we just need to initialize the network weights with the default
initialization method, which draws random values uniformly from $[-0.7, 0.7]$.
You can see this in the following example.

```{.python .input  n=5}
layer.initialize()
```

Now you can pass your data through a network. Passing a data through a network
is also called a forward pass. Do a forward pass with random data, shown in the
following example. We create a $(10,3)$ shape random input `x` and feed into the
layer to compute the output.

```{.python .input  n=6}
x = np.random.uniform(-1,1,(10,3))
layer(x)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "array([[-0.        , -0.        ,  0.04437108,  0.01096384, -0.        ],\n       [ 0.01540359,  0.01735983, -0.        , -0.        ,  0.03530429],\n       [ 0.00381139,  0.02891595,  0.0071606 ,  0.01262893, -0.        ],\n       [ 0.02378628,  0.04612946, -0.        , -0.        ,  0.05157538],\n       [-0.        ,  0.00375629, -0.        , -0.        ,  0.00060787],\n       [-0.        , -0.        ,  0.01947428,  0.00730047, -0.        ],\n       [ 0.03439108,  0.04268821,  0.02091249, -0.        ,  0.03359453],\n       [ 0.01674652,  0.05354869, -0.        , -0.        ,  0.05611669],\n       [ 0.01591048,  0.04680086,  0.05175146,  0.03321034, -0.        ],\n       [ 0.00959048,  0.03363482,  0.03384075,  0.02388959, -0.        ]])"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The layer produced a $(10,5)$ shape output from our $(10,3)$ input.

**When you dont specify the `in_unit` parameter, the system  automatically
infers it during the first time you feed in data during the first forward step
after you create and initialize the weights.**

You can access the weight after the first forward pass, as shown in this
example.

```{.python .input  n=7}
layer.weight.data()
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "array([[ 0.0068339 ,  0.01299825,  0.0301265 ],\n       [ 0.04819721,  0.01438687,  0.05011239],\n       [ 0.00628365,  0.04861524, -0.01068833],\n       [ 0.01729892,  0.02042518, -0.01618656],\n       [-0.00873779, -0.02834515,  0.05484822]])"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.python .input  n=8}
net = nn.Sequential()

net.add(nn.Dense(5,in_units=3,activation='relu'),
        nn.Dense(25, activation='relu'),
        nn.Dense(2)
       )

net
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "Sequential(\n  (0): Dense(3 -> 5, Activation(relu))\n  (1): Dense(-1 -> 25, Activation(relu))\n  (2): Dense(-1 -> 2, linear)\n)"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The layers are ordered exactly the way you defined your neural network with
numbers starting from 0. You can access the layers by indexing the network using
`[]`.

```{.python .input  n=9}
net[1]
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "Dense(-1 -> 25, Activation(relu))"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.python .input  n=10}
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

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "MLP(\n  (dense1): Dense(-1 -> 5, Activation(relu))\n  (dense2): Dense(-1 -> 25, Activation(relu))\n  (dense3): Dense(-1 -> 2, linear)\n)"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Simirly, you can the following code to implement a famous network called
[LeNet](http://yann.lecun.com/exdb/lenet/) through `nn.Sequential`.

```{.python .input  n=11}
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
    
net = LeNet()
net
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "LeNet(\n  (conv1): Conv2D(-1 -> 6, kernel_size=(3, 3), stride=(1, 1), Activation(relu))\n  (pool1): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)\n  (conv2): Conv2D(-1 -> 16, kernel_size=(3, 3), stride=(1, 1), Activation(relu))\n  (pool2): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)\n  (dense1): Dense(-1 -> 120, Activation(relu))\n  (dense2): Dense(-1 -> 84, Activation(relu))\n  (dense3): Dense(-1 -> 10, linear)\n)"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=12}
image_data = np.random.uniform(-1,1, (1,1,28,28))

net.initialize()

net(image_data)
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "array([[ 0.00052316, -0.00139668,  0.00031271,  0.00159006, -0.00019339,\n         0.00017942,  0.00140935, -0.00048231, -0.00016249,  0.00099462]])"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In this example, you can use `print` to get the intermediate results between
layers. This approach provides a more flexible way to define the forward
function.

You can use `[]` to index a particular layer. For example, the following
accesses the first layer's weight and sixth layer's bias.

```{.python .input  n=13}
net.conv1.weight.data().shape, net.dense1.bias.data().shape

```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "((6, 1, 3, 3), (120,))"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.python .input  n=18}
from mxnet.gluon import model_zoo

net = model_zoo.vision.resnet50_v2(pretrained=True)
net.hybridize()

dummy_input = np.ones(shape=(1,3,224,224))
output = net(dummy_input)
output.shape
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "(1, 1000)"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Deciding the paradigm for your NN

MXNet includes two types of programming paradigms:

1. Imperative API - Gluon that is very user friendly and allows for quick
prototyping, easy debugging and natural control flow for people familiar with
python programming.
2. Symbolic or Declarative API for creating static graphs with low level
optimizations on operators and allows for support in multiple languages, cross-
compatibility with other frameworks (via ONNX). However, it’s less flexible
because any logic must be encoded into the graph as special operators like scan,
while_loop and cond. It’s also hard to debug.

So how can we make use of Symbolic API while getting the flexibility of an
Imperative API to quickly protype and debug??

Enter **HybridBlocks**

Each HybridBlock can run fully imperatively defining their computation with real
functions acting on real inputs. But they’re also capable of running
symbolically, acting on placeholders. Gluon hides most of this under the hood so
you’ll only need to know how it works when you want to write your own layers.

```{.python .input  n=19}
net_hybrid_seq = nn.HybridSequential()

net_hybrid_seq.add(nn.Dense(5,in_units=3,activation='relu'),
        nn.Dense(25, activation='relu'),
        nn.Dense(2)
       )

net_hybrid_seq
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "HybridSequential(\n  (0): Dense(3 -> 5, Activation(relu))\n  (1): Dense(-1 -> 25, Activation(relu))\n  (2): Dense(-1 -> 2, linear)\n)"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

To compile and optimize the `HybridSequential`, we can then call its `hybridize
method`.

```{.python .input  n=20}
net_hybrid_seq.hybridize()
```

Performance

To get a sense of the speedup from hybridizing, we can compare the performance
before and after hybridizing by measuring in either case the time it takes to
make 1000 forward passes through the network.

```{.python .input  n=21}
from time import time

def bench(net, x):
    y = net(x)
    start = time()
    for i in range(1,1000):
        y = net(x)
    return time() - start

x = np.random.normal(size=(1,512))

net_hybrid_seq = nn.HybridSequential()

net_hybrid_seq.add(nn.Dense(256,activation='relu'),
        nn.Dense(128, activation='relu'),
        nn.Dense(2)
       )
net_hybrid_seq.initialize()

print('Before hybridizing: %.4f sec'%(bench(net_hybrid_seq, x)))
net_hybrid_seq.hybridize()
print('After hybridizing: %.4f sec'%(bench(net_hybrid_seq, x)))
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before hybridizing: 0.6364 sec\nAfter hybridizing: 0.1622 sec\n"
 }
]
```

Peeling another layer, we also have a `HybridBlock` which is the hybrid version
of the `Block` API.

With normal Blocks, we just need to define a `forward` function that takes an
input `x` and computes the result of the forward pass through the network. To
define a `HybridBlock`, we create the same `forward` function. MXNet takes care
of hybridizing the model at the backend so you don't have to make changes to
your code to convert it to a symbolic paradigm

```{.python .input  n=152}
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
        return x
    
net_Hybrid = MLP_Hybrid()
net_Hybrid.initialize()

print('Before hybridizing: %.4f sec'%(bench(net_Hybrid, x)))
net_Hybrid.hybridize()
print('After hybridizing: %.4f sec'%(bench(net_hybrid_seq, x)))
```

```{.json .output n=152}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before hybridizing: 0.6658 sec\nAfter hybridizing: 0.2191 sec\n"
 }
]
```

Given a HybridBlock whose forward computation consists of going through other
HybridBlocks, you can compile that section of the network by calling the
HybridBlocks `.hybridize()` method.

All of MXNet’s predefined layers are HybridBlocks. This means that any network
consisting entirely of predefined MXNet layers can be compiled and run at much
faster speeds by calling `.hybridize()`.

## Next steps: TODO: UPDATE

After you create a neural network, learn how to automatically
compute the gradients in [Step 3: Automatic differentiation with
autograd](3-autograd.md).

```{.python .input}

```
