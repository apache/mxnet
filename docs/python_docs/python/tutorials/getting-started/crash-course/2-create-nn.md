# Step 2: Create a neural network

In this step, you learn how to use NP on MXNet to create neural networks in
Gluon. In addition to the `np` package that you learned about in the previous
step [Step 1: Manipulate data with NP on MXNet](1-ndarray.md), you also import
the neural networks from `gluon`. Gluon contains large number of build-in neural
network layers in the following two modules:

1. `mxnet.gluon.nn`
2. `mxnet.gluon.contrib.nn`

Use the following commands to import the packages required for this step.

```{.python .input  n=52}
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

```{.python .input  n=19}
layer = nn.Dense(5)
layer
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "Dense(-1 -> 5, linear)"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In the example above, the **-1** denotes that the size of the input layer is not
specified during initialization.

You can also call the **Dense** layer with an `in_units` parameter if you know
the shape of your input unit.

```{.python .input  n=22}
layer = nn.Dense(5,in_units=3)
layer
```

```{.json .output n=22}
[
 {
  "data": {
   "text/plain": "Dense(3 -> 5, linear)"
  },
  "execution_count": 22,
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

```{.python .input  n=37}
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

```{.python .input  n=38}
layer.initialize()
```

Now you can pass your data through a network. Passing a data through a network
is also called a forward pass. Do a forward pass with random data, shown in the
following example. We create a $(10,3)$ shape random input `x` and feed into the
layer to compute the output.

```{.python .input  n=40}
x = np.random.uniform(-1,1,(10,3))
layer(x)
```

```{.json .output n=40}
[
 {
  "data": {
   "text/plain": "array([[0.        , 0.        , 0.02514516, 0.03788552, 0.        ],\n       [0.00586026, 0.05775406, 0.        , 0.        , 0.03826544],\n       [0.01782359, 0.02893867, 0.        , 0.        , 0.02829976],\n       [0.02690828, 0.        , 0.        , 0.        , 0.00545937],\n       [0.03234754, 0.        , 0.02270631, 0.02974202, 0.        ],\n       [0.00555724, 0.01362655, 0.        , 0.        , 0.01144097],\n       [0.        , 0.04869641, 0.        , 0.        , 0.        ],\n       [0.        , 0.        , 0.00969722, 0.00794899, 0.        ],\n       [0.01007228, 0.        , 0.02497568, 0.03668693, 0.        ],\n       [0.        , 0.        , 0.04090203, 0.05625763, 0.        ]])"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

As can be seen, the layer's input limit of two produced a $(10,5)$ shape output
from our $(10,3)$ input. When you dont specify the `in_unit` parameter, the
system  automatically infers it during the first time you feed in data, create,
and initialize the weights. You can access the weight after the first forward
pass, as shown in this example.

```{.python .input  n=41}
layer.weight.data()
```

```{.json .output n=41}
[
 {
  "data": {
   "text/plain": "array([[ 0.0252179 ,  0.0193032 , -0.06099349],\n       [-0.04770225, -0.05782406,  0.02741051],\n       [ 0.01796688,  0.01117834,  0.02134723],\n       [ 0.02854334,  0.00740966,  0.03457288],\n       [ 0.00836869, -0.02033399, -0.0502243 ]])"
  },
  "execution_count": 41,
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

```{.python .input  n=47}
net = nn.Sequential()

net.add(nn.Dense(5,in_units=3,activation='relu'),
        nn.Dense(25, activation='relu'),
        nn.Dense(2)
       )

net
```

```{.json .output n=47}
[
 {
  "data": {
   "text/plain": "Sequential(\n  (0): Dense(3 -> 5, Activation(relu))\n  (1): Dense(-1 -> 25, Activation(relu))\n  (2): Dense(-1 -> 2, linear)\n)"
  },
  "execution_count": 47,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The layers are ordered exactly the way you defined your neural network with
numbers starting from 0. You can access the layers by indexing the network using
`[]`.

```{.python .input  n=49}
net[1]
```

```{.json .output n=49}
[
 {
  "data": {
   "text/plain": "Dense(-1 -> 25, Activation(relu))"
  },
  "execution_count": 49,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Create a custom neural network architecture flexibly

`nn.Sequential()` allows you to create your multi-layer neural network with
existing layers from `gluon.nn`. It also includes a pre-defined `forward()`
function that sequentially execututes added layers. But what if we want to add
more computation during your forward pass or createa  new/novel network. How do
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
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)

    def forward(self, x):
        pass
```

```{.python .input  n=54}
class MLP(Block):
    def __init__(self):
        super(MLP, self).__init__()
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

```{.json .output n=54}
[
 {
  "data": {
   "text/plain": "MLP(\n  (dense1): Dense(-1 -> 5, Activation(relu))\n  (dense2): Dense(-1 -> 25, Activation(relu))\n  (dense3): Dense(-1 -> 2, linear)\n)"
  },
  "execution_count": 54,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Simirly, you can the following code to implement a famous network called
[LeNet](http://yann.lecun.com/exdb/lenet/) through `nn.Sequential`.

```{.python .input  n=120}
class LeNet(Block):
    def __init__(self):
        super(LeNet, self).__init__()
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

```{.json .output n=120}
[
 {
  "data": {
   "text/plain": "LeNet(\n  (conv1): Conv2D(-1 -> 6, kernel_size=(3, 3), stride=(1, 1), Activation(relu))\n  (pool1): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)\n  (conv2): Conv2D(-1 -> 16, kernel_size=(3, 3), stride=(1, 1), Activation(relu))\n  (pool2): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)\n  (dense1): Dense(-1 -> 120, Activation(relu))\n  (dense2): Dense(-1 -> 84, Activation(relu))\n  (dense3): Dense(-1 -> 10, linear)\n)"
  },
  "execution_count": 120,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=102}
image_data = np.random.uniform(-1,1, (1,1,28,28))

net.initialize()

net(image_data)
```

```{.json .output n=102}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/mxnet/gluon/block.py:571: UserWarning: Parameter 'weight' is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  v.initialize(None, ctx, init, force_reinit=force_reinit)\n/home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/mxnet/gluon/block.py:571: UserWarning: Parameter 'bias' is already initialized, ignoring. Set force_reinit=True to re-initialize.\n  v.initialize(None, ctx, init, force_reinit=force_reinit)\n"
 },
 {
  "data": {
   "text/plain": "array([[ 9.5244788e-05,  2.9054983e-03, -6.9242873e-04,  2.2817445e-03,\n        -8.9324836e-04, -6.6934002e-04, -2.7392569e-03,  3.3438532e-04,\n        -4.0528120e-04,  2.8272711e-03]])"
  },
  "execution_count": 102,
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

```{.python .input  n=108}
net.conv1.weight.data().shape, net.dense1.bias.data().shape

```

```{.json .output n=108}
[
 {
  "data": {
   "text/plain": "((6, 1, 3, 3), (120,))"
  },
  "execution_count": 108,
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

```{.python .input  n=117}
from mxnet.gluon import model_zoo

net = model_zoo.vision.resnet50_v2(pretrained=True)
net.hybridize()

dummy_input = np.ones(shape=(1,3,224,224))
output = net(dummy_input)

output.shape
        
```

```{.json .output n=117}
[
 {
  "data": {
   "text/plain": "(1, 1000)"
  },
  "execution_count": 117,
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
compatibility with other frameworks (via ONNX) and as a backend for other high
level APIs like keras. However, it’s less flexible because any logic must be
encoded into the graph as special operators like scan, while_loop and cond. It’s
also hard to debug.

So how can we make use of Symbolic API while getting the flexibility of an
Imperative API to quickly protype and debug??

Enter **HybridBlocks**

Each HybridBlock can run fully imperatively defining their computation with real
functions acting on real inputs. But they’re also capable of running
symbolically, acting on placeholders. Gluon hides most of this under the hood so
you’ll only need to know how it works when you want to write your own layers.

```{.python .input  n=122}
net_hybrid_seq = nn.HybridSequential()

net_hybrid_seq.add(nn.Dense(5,in_units=3,activation='relu'),
        nn.Dense(25, activation='relu'),
        nn.Dense(2)
       )

net_hybrid_seq
```

```{.json .output n=122}
[
 {
  "data": {
   "text/plain": "HybridSequential(\n  (0): Dense(3 -> 5, Activation(relu))\n  (1): Dense(-1 -> 25, Activation(relu))\n  (2): Dense(-1 -> 2, linear)\n)"
  },
  "execution_count": 122,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

To compile and optimize the `HybridSequential`, we can then call its `hybridize
method`.

```{.python .input  n=124}
net_hybrid_seq.hybridize()
```

Performance

To get a sense of the speedup from hybridizing, we can compare the performance
before and after hybridizing by measuring in either case the time it takes to
make 1000 forward passes through the network.

```{.python .input  n=137}
from time import time

def bench(net, x):
    start = time()
    for i in range(1000):
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

```{.json .output n=137}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Before hybridizing: 0.6753 sec\nAfter hybridizing: 0.2139 sec\n"
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
        super(MLP_Hybrid, self).__init__()
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

## Next steps

After you create a neural network, learn how to automatically
compute the gradients in [Step 3: Automatic differentiation with
autograd](3-autograd.md).

```{.python .input}

```
