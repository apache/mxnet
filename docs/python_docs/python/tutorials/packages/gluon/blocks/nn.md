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

# Layers and Blocks

<!-- adapted from diveintodeeplearning -->

As network complexity increases, we move from designing single to entire layers
of neurons.

Neural network designs like
[ResNet-152](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
have a fair degree of regularity. They consist of *blocks* of repeated (or at
least similarly designed) layers; these blocks then form the basis of more
complex network designs.

In this section, we'll talk about how to write code that makes such blocks on
demand, just like a Lego factory generates blocks which can be combined to
produce terrific artifacts.

We start with a very simple block, namely the block for a multilayer
perceptron. A common strategy would be to construct a two-layer network as
follows:

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn


x = nd.random.uniform(shape=(2, 20))

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

This generates a network with a hidden layer of $256$ units, followed by a ReLU
activation and another $10$ units governing the output. In particular, we used
the [nn.Sequential](/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Sequential)
constructor to generate an empty network into which we then inserted both
layers. What exactly happens inside `nn.Sequential`
has remained rather mysterious so far. In the following we will see that this
really just constructs a block that is a container for other blocks. These
blocks can be combined into larger artifacts, often recursively. The diagram
below shows how:

![Blocks can be used recursively to form larger artifacts](blocks.svg)

In the following we will explain the various steps needed to go from defining
layers to defining blocks (of one or more layers):

1. Blocks take data as input.
1. Blocks store state in the form of parameters that are inherent to the block.
   For instance, the block above contains two hidden layers, and we need a
   place to store parameters for it.
1. Blocks produce meaningful output. This is typically encoded in what
   we will call the `forward` function. It allows us to invoke a block via
   `net(X)` to obtain the desired output. What happens behind the scenes is
   that it invokes `forward` to perform forward propagation (also called
   forward computation).
1. Blocks initialize the parameters in a lazy fashion as part of the first
   `forward` call.
1. Blocks calculate a gradient with regard to their input when invoking
   `backward`. Typically this is automatic.

## A Sequential Block

The [Block](/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Block) class is a generic component
describing data flow. When the data flows through a sequence of blocks, each
block applied to the output of the one before with the first block being
applied on the input data itself, we have a special kind of block, namely the
`Sequential` block.

`Sequential` has helper methods to manage the sequence, with `add` being the
main one of interest allowing you to append blocks in sequence. Once the
operations have been added, the forward computation of the model applies the
blocks on the input data in the order they were added.  Below, we implement a
`MySequential` class that has the same functionality as the `Sequential` class.
This may help you understand more clearly how the `Sequential` class works.

```{.python .input  n=3}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume it has a unique name. We save it in the
        # member variable _children of the Block class, and its type is OrderedDict. When the MySequential instance
        # calls the initialize function, the system automatically initializes all members of _children.
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order they were added.
        for block in self._children.values():
            x = block(x)
        return x
```

At its core is the `add` method. It adds any block to the ordered dictionary of
children. These are then executed in sequence when forward propagation is
invoked. Let's see what the MLP looks like now.

```{.python .input  n=4}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

Indeed, it is no different than It can observed here that the use of the
`MySequential` class is no different from the use of the Sequential class.


## A Custom Block

It is easy to go beyond simple concatenation with `Sequential`. The
`Block` class provides the functionality required to make such customizations.
`Block` has a model constructor provided in the `nn` module, which we can
inherit to define the model we want. The following inherits the `Block` class to
construct the multilayer perceptron mentioned at the beginning of this section.
The `MLP` class defined here overrides the `__init__` and `forward` functions
of the Block class. They are used to create model parameters and define forward
computations, respectively. Forward computation is also forward propagation.

```{.python .input  n=1}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers.

    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections.
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.output = nn.Dense(10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input x.

    def forward(self, x):
        hidden_out = self.hidden(x)
        return self.output(hidden_out)
```

Let's look at it a bit more closely. The `forward` method invokes a network
simply by evaluating the hidden layer `self.hidden(x)` and subsequently by
evaluating the output layer `self.output( ... )`. This is what we expect in the
forward pass of this block.

In order for the block to know what it needs to evaluate, we first need to
define the layers. This is what the `__init__` method does. It first
initializes all of the Block-related parameters and then constructs the
requisite layers. This attaches the coresponding layers and the required
parameters to the class. Note that there is no need to define a backpropagation
method in the class. The system automatically generates the `backward` method
needed for back propagation by automatically finding the gradient (see the tutorial on [autograd](/api/python/docs/tutorials/packages/autograd/index.html)). The same
applies to the [initialize](/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Block.initialize) method, which is generated automatically. Let's try
this out:

```{.python .input  n=2}
net = MLP()
net.initialize()
net(x)
```

As explained above, the `Block` class can be quite versatile in terms of what it
does. For instance, its subclass can be a layer (such as the `Dense` class
provided by Gluon), it can be a model (such as the `MLP` class we just derived),
or it can be a part of a model (this is what typically happens when designing
very deep networks). Throughout this chapter we will see how to use this with
great flexibility.


## Coding with `Blocks`

### Blocks
The [Sequential](/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Sequential) class
can make model construction easier and does not require you to define the
`forward` method; however, directly inheriting from
its parent class, [Block](/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Block), can greatly
expand the flexibility of model construction. For example, implementing the
`forward` method means you can introduce control flow in the network.

### Constant parameters
Now we'd like to introduce the notation of a *constant* parameter. These are
parameters that are not used when invoking backpropagation. This sounds very
abstract but here's what's really going on.
Assume that we have some function

$$f(\mathbf{x},\mathbf{w}) = 3 \cdot \mathbf{w}^\top \mathbf{x}.$$

In this case $3$ is a constant parameter. We could change $3$ to something else,
say $c$ via

$$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}.$$

Nothing has really changed, except that we can adjust the value of $c$. It is
still a constant as far as $\mathbf{w}$ and $\mathbf{x}$ are concerned. However,
Gluon doesn't know about this unless we create it with `get_constant`
(this makes the code go faster, too, since we're not sending the Gluon engine
on a wild goose chase after a parameter that doesn't change).

```{.python .input  n=5}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        # Random weight parameters created with the get_constant are not
        # iterated during training (i.e. constant parameters).
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # Use the constant parameters created, as well as the ReLU and dot
        # functions of NDArray.

        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # Re-use the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers.
        x = self.dense(x)
        # Here in the control flow, we need to call `asscalar` to return the
        # scalar for comparison.

        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
```

In this `FancyMLP` model, we used constant weight `rand_weight` (note that it is
not a model parameter), performed a matrix multiplication operation (`nd.dot`),
and reused the *same* `Dense` layer. Note that this is very different from using
two dense layers with different sets of parameters. Instead, we used the same
network twice. Quite often in deep networks one also says that the parameters
are *tied* when one wants to express that multiple parts of a network share the
same parameters. Let's see what happens if we construct it and feed data through
it.

```{.python .input  n=6}
net = FancyMLP()
net.initialize()
net(x)
```

There's no reason why we couldn't mix and match these ways of building a
network. Obviously the example below resembles a [Rube Goldberg
Machine](https://en.wikipedia.org/wiki/Rube_Goldberg_machine). That said, it
combines examples for building a block from individual blocks,
which in turn, may be blocks themselves. Furthermore, we can even combine
multiple strategies inside the same forward function. To demonstrate this,
here's the network.

```{.python .input  n=7}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FancyMLP())

chimera.initialize()
chimera(x)
```

## Hybridization

The reader may be starting to think about the efficiency of this Python code.
After all, we have lots of dictionary lookups, code execution, and lots of
other Pythonic things going on in what is supposed to be a high performance
deep learning library. The problems of Python's [Global Interpreter
Lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well
known.

In the context of deep learning, we often have highly performant GPUs that
depend on CPUs running Python to tell them what to do. This mismatch can
manifest in the form of GPU starvation when the CPUs can not provide
instruction fast enough. We can improve this situation by deferring to a more
performant language instead of Python when possible.

Gluon does this by allowing for [Hybridization](hybridize.html). In it, the
Python interpreter executes the block the first time it's invoked. The Gluon
runtime records what is happening and the next time around it short circuits
any calls to Python. This can accelerate things considerably in some cases but
care needs to be taken with [control flow](/api/python/docs/tutorials/packages/autograd/index.html#Advanced:-Using-Python-control-flow).
