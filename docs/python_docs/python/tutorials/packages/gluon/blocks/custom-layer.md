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

While Gluon API for Apache MxNet comes with [a decent number of pre-defined layers](https://mxnet.apache.org/api/python/gluon/nn.html), at some point one may find that a new layer is needed. Adding a new layer in Gluon API is straightforward, yet there are a few things that one needs to keep in mind.

In this article, I will cover how to create a new layer from scratch, how to use it, what are possible pitfalls and how to avoid them.

## The simplest custom layer

To create a new layer in Gluon API, one must create a class that inherits from [Block](https://github.com/apache/incubator-mxnet/blob/c9818480680f84daa6e281a974ab263691302ba8/python/mxnet/gluon/block.py#L128) class. This class provides the most basic functionality, and all pre-defined layers inherit from it directly or via other subclasses. Because each layer in Apache MxNet inherits from `Block`, words "layer" and "block" are used interchangeable inside of the Apache MxNet community.

The only instance method needed to be implemented is [forward(self, x)](https://github.com/apache/incubator-mxnet/blob/c9818480680f84daa6e281a974ab263691302ba8/python/mxnet/gluon/block.py#L909), which defines what exactly your layer is going to do during forward propagation. Notice, that it doesn't require to provide what the block should do during back propogation. Back propogation pass for blocks is done by Apache MxNet for you. 

In the example below, we define a new layer and implement `forward()` method to normalize input data by fitting it into a range of [0, 1].


```{.python .input}
# Do some initial imports used throughout this tutorial 
from __future__ import print_function
import mxnet as mx
from mxnet import np, npx, gluon, autograd
from mxnet.gluon.nn import Dense
mx.random.seed(1)                      # Set seed for reproducable results
```


```{.python .input}
class NormalizationLayer(gluon.Block):
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
```

The rest of methods of the `Block` class are already implemented, and majority of them are used to work with parameters of a block. There is one very special method named [hybridize()](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/block.py#L384), though, which I am going to cover before moving to a more complex example of a custom layer.

## Hybridization and the difference between Block and HybridBlock

Looking into implementation of [existing layers](https://mxnet.apache.org/api/python/gluon/nn.html), one may find that more often a block inherits from a [HybridBlock](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/block.py#L428), instead of directly inheriting from `Block`.

The reason for that is that `HybridBlock` allows to write custom layers in imperative programming style, while computing in a symbolic way. It unifies the flexibility of imperative programming with the performance benefits of symbolic programming. You can learn more about the difference between symbolic and imperative programming from [this article](https://mxnet.apache.org/api/architecture/overview.html).

Hybridization is a process that Apache MxNet uses to create a symbolic graph of a forward computation. This allows to increase computation performance by optimizing the computational symbolic graph. Once the symbolic graph is created, Apache MxNet caches and reuses it for subsequent computations.

Hybridization of HybridBlock.forward is based on a deferred computation mode in the MXNet backend, which enables recording computation via tracing in the mxnet.nd and mxnet.np interfaces. The recorded computation can be exported to a symbolic representation and is used for optimized execution with the CachedOp.

As tracing is based on the imperative APIs, users can access shape information of the arrays. As x.shape for some array x is a python tuple, any use of that shape will be a constant in the recorded graph and may limit the recorded graph to be used with inputs of the same shape only.

Knowing this, we can rewrite our example layer, using HybridBlock:


```{.python .input}
class NormalizationHybridLayer(gluon.HybridBlock):
    def __init__(self):
        super(NormalizationHybridLayer, self).__init__()

    def forward(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
```

Thanks to inheriting from HybridBlock, one can easily do forward pass on a given ndarray, either on CPU or GPU:


```{.python .input}
layer = NormalizationHybridLayer()
layer(np.array([1, 2, 3], ctx=mx.cpu()))
```

Output:

```bash
[0.  0.5 1. ]
```


As a rule of thumb, one should always implement custom layers by inheriting from `HybridBlock`. This allows to have more flexibility, and doesn't affect execution speed once hybridization is done. 

Unfortunately, at the moment of writing this tutorial, NLP related layers such as [RNN](https://mxnet.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.RNN), [GRU](https://mxnet.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.GRU) and [LSTM](https://mxnet.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.LSTM) are directly inhereting from the `Block` class via common `_RNNLayer` class. That means that networks with such layers cannot be hybridized. But this might change in the future, so stay tuned.

It is important to notice that hybridization has nothing to do with computation on GPU. One can train both hybridized and non-hybridized networks on both CPU and GPU, though hybridized networks would work faster. Though, it is hard to say in advance how much faster it is going to be.

## Adding a custom layer to a network

While it is possible, custom layers are rarely used separately. Most often they are used with predefined layers to create a neural network. Output of one layer is used as an input of another layer.

Depending on which class you used as a base one, you can use either [Sequential](https://mxnet.apache.org/api/python/gluon/gluon.html#mxnet.gluon.nn.Sequential) or [HybridSequential](https://mxnet.apache.org/api/python/gluon/gluon.html#mxnet.gluon.nn.HybridSequential) container to form a sequential neural network. By adding layers one by one, one adds dependencies of one layer's input from another layer's output. It is worth noting, that both `Sequential` and `HybridSequential` containers inherit from `Block` and `HybridBlock` respectively. 

Below is an example of how to create a simple neural network with a custom layer. In this example, `NormalizationHybridLayer` gets as an input the output from `Dense(5)` layer and pass its output as an input to `Dense(1)` layer.


```{.python .input}
net = gluon.nn.HybridSequential()                         # Define a Neural Network as a sequence of hybrid blocks
net.add(Dense(5))                                     # Add Dense layer with 5 neurons
net.add(NormalizationHybridLayer())                   # Add a custom layer
net.add(Dense(1))                                     # Add Dense layer with 1 neurons


net.initialize(mx.init.Xavier(magnitude=2.24))            # Initialize parameters of all layers
net.hybridize()                                           # Create, optimize and cache computational graph
input = np.random.uniform(low=-10, high=10, size=(5, 2))  # Create 5 random examples with 2 feature each in range [-10, 10]
net(input)
```


Output:

```bash
[[-0.13601446]
 [ 0.26103732]
 [-0.05046433]
 [-1.2375476 ]
 [-0.15506986]]
```


## Parameters of a custom layer

Usually, a layer has a set of associated parameters, sometimes also referred as weights. This is an internal state of a layer. Most often, these parameters are the ones, that we want to learn during backpropogation step, but sometimes these parameters might be just constants we want to use during forward pass. The parameters are usually represented as [Parameter](https://mxnet.apache.org/api/python/gluon/gluon.html#mxnet.gluon.Parameter) class inside of Apache MXNet neural network.


```{.python .input}
class NormalizationHybridLayer(gluon.HybridBlock):
    def __init__(self, hidden_units, scales):
        super(NormalizationHybridLayer, self).__init__()
        self.hidden_units = hidden_units
        self.weights = gluon.Parameter('weights',
                                       shape=(hidden_units, -1),
                                       allow_deferred_init=True)

        self.scales = gluon.Parameter('scales',
                                      shape=scales.shape,
                                      init=mx.init.Constant(scales), # Convert to regular list to make this object serializable
                                      differentiable=False)
            
    def forward(self, x):
        normalized_data = (x - np.min(x)) / (np.max(x) - np.min(x))
        weighted_data = npx.fully_connected(normalized_data, self.weights.data(), num_hidden=self.hidden_units, no_bias=True)
        scaled_data = np.multiply(self.scales.data(), weighted_data)
        return scaled_data
    
    def infer_shape(self, x, *args):
        self.weights.shape = (self.hidden_units, x.shape[x.ndim-1])
```

In the example above 2 set of parameters are defined:
1. Parameter `weights` is trainable. Its shape is unknown during construction phase and will be infered on the first run of forward propogation; 
1. Parameter `scale` is a constant that doesn't change. Its shape is defined during construction.

Notice a few aspects of this code:
* Shape is not provided when creating `weights`. Instead it is going to be infered from the shape of the input by `infer_shape` method.
* `Scales` parameter is initialized and marked as `differentiable=False`.

Running forward pass on this network is very similar to the previous example, so instead of just doing one forward pass, let's run whole training for a few epochs to show that `scales` parameter doesn't change during the training while `weights` parameter is changing.


```{.python .input}
def print_params(title, net):
    """
    Helper function to print out the state of parameters of NormalizationHybridLayer
    """
    print(title)
    hybridlayer_params = {k: v for k, v in net.collect_params().items()}
    
    for key, value in hybridlayer_params.items():
        print('{} = {}\n'.format(key, value.data()))

net = gluon.nn.HybridSequential()                             # Define a Neural Network as a sequence of hybrid blocks
net.add(Dense(5))                                         # Add Dense layer with 5 neurons
net.add(NormalizationHybridLayer(hidden_units=5, 
                                 scales = np.array([2]))) # Add a custom layer
net.add(Dense(1))                                         # Add Dense layer with 1 neurons


net.initialize(mx.init.Xavier(magnitude=2.24))                # Initialize parameters of all layers
net.hybridize()                                               # Create, optimize and cache computational graph

input = np.random.uniform(low=-10, high=10, size=(5, 2))      # Create 5 random examples with 2 feature each in range [-10, 10]
label = np.random.uniform(low=-1, high=1, size=(5, 1))

mse_loss = gluon.loss.L2Loss()                                # Mean squared error between output and label
trainer = gluon.Trainer(net.collect_params(),                 # Init trainer with Stochastic Gradient Descent (sgd) optimization method and parameters for it
                        'sgd', 
                        {'learning_rate': 0.1, 'momentum': 0.9 })
                        
with autograd.record():                                       # Autograd records computations done on NDArrays inside "with" block 
    output = net(input)                                       # Run forward propogation
    
    print_params("=========== Parameters after forward pass ===========\n", net)    
    loss = mse_loss(output, label)                            # Calculate MSE
    
loss.backward()                                               # Backward computes gradients and stores them as a separate array within each NDArray in .grad field
trainer.step(input.shape[0])                                  # Trainer updates parameters of every block, using .grad field using oprimization method (sgd in this example)
                                                              # We provide batch size that is used as a divider in cost function formula
print_params("=========== Parameters after backward pass ===========\n", net)
```

Output:

```bash
=========== Parameters after forward pass ===========

hybridsequential94_normalizationhybridlayer0_weights = 
[[-0.3983642  -0.505708   -0.02425683 -0.3133553  -0.35161012]
 [ 0.6467543   0.3918715  -0.6154656  -0.20702496 -0.4243446 ]
 [ 0.6077331   0.03922009  0.13425875  0.5729856  -0.14446527]
 [-0.3572498   0.18545026 -0.09098256  0.5106366  -0.35151464]
 [-0.39846328  0.22245121  0.13075739  0.33387476 -0.10088372]]

hybridsequential94_normalizationhybridlayer0_scales = 
[2.]

=========== Parameters after backward pass ===========

hybridsequential94_normalizationhybridlayer0_weights = 
[[-0.29839832 -0.47213346  0.08348035 -0.2324698  -0.27368504]
 [ 0.76268613  0.43080837 -0.49052125 -0.11322092 -0.3339738 ]
 [ 0.48665082 -0.00144657  0.00376363  0.47501418 -0.23885089]
 [-0.22626656  0.22944227  0.05018325  0.6166192  -0.24941102]
 [-0.44946212  0.20532274  0.07579394  0.29261002 -0.14063817]]

hybridsequential94_normalizationhybridlayer0_scales = 
[2.]
``` 


As it is seen from the output above, `weights` parameter has been changed by the training and `scales` not.

## Conclusion

One important quality of a Deep learning framework is extensibility. Empowered by flexible abstractions, like `Block` and `HybridBlock`, one can easily extend Apache MxNet functionality to match its needs.
