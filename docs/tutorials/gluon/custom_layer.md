
# How to write a custom layer in Apache MxNet Gluon API

While Gluon API for Apache MxNet comes with [a decent number of pre-defined layers](https://mxnet.incubator.apache.org/api/python/gluon/nn.html), at some point one may find that a new layer is needed. Adding a new layer in Gluon API is straightforward, yet there are a few things that one needs to keep in mind.

In this article, I will cover how to create a new layer from scratch, how to use it, what are possible pitfalls and how to avoid them.

## The simplest custom layer

To create a new layer in Gluon API, one must create a class that inherits from [Block](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/block.py#L123) class. This class provides the most basic functionality, and all pre-defined layers inherit from it directly or via other subclasses. Because each layer in Apache MxNet inherits from `Block`, words "layer" and "block" are used interchangeable inside of the Apache MxNet community.

The only instance method needed to be implemented is [forward(self, x)](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/block.py#L415), which defines what exactly your layer is going to do during forward propagation. Notice, that it doesn't require to provide what the block should do during back propogation. Back propogation pass for blocks is done by Apache MxNet for you. 

In the example below, we define a new layer and implement `forward()` method to normalize input data by fitting it into a range of [0, 1].


```python
# Do some initial imports used throughout this tutorial 
from __future__ import print_function
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.nn import Dense
mx.random.seed(1)                      # Set seed for reproducable results
```


```python
class NormalizationLayer(gluon.Block):
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x):
        return (x - nd.min(x)) / (nd.max(x) - nd.min(x))
```

The rest of methods of the `Block` class are already implemented, and majority of them are used to work with parameters of a block. There is one very special method named [hybridize()](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/block.py#L384), though, which I am going to cover before moving to a more complex example of a custom layer.

## Hybridization and the difference between Block and HybridBlock

Looking into implementation of [existing layers](https://mxnet.incubator.apache.org/api/python/gluon/nn.html), one may find that more often a block inherits from a [HybridBlock](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/block.py#L428), instead of directly inheriting from `Block`.

The reason for that is that `HybridBlock` allows to write custom layers that can be used in imperative programming as well as in symbolic programming. It is convinient to support both ways, because the imperative programming eases the debugging of the code and the symbolic one provides faster execution speed. You can learn more about the difference between symbolic vs. imperative programming from [this article](https://mxnet.incubator.apache.org/architecture/program_model.html).

Hybridization is a process that Apache MxNet uses to create a symbolic graph of a forward computation. This allows to increase computation performance by optimizing the computational symbolic graph. Once the symbolic graph is created, Apache MxNet caches and reuses it for subsequent computations.

To simplify support of both imperative and symbolic programming, Apache MxNet introduce the `HybridBlock` class. Compare to the `Block` class, `HybridBlock` already has its [forward()](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.HybridBlock.forward) method implemented, but it defines a [hybrid_forward()](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.HybridBlock.hybrid_forward) method that needs to be implemented.

The main difference between `forward()` and `hybrid_forward()` is an `F` argument. This argument sometimes is refered as a `backend` in the Apache MxNet community. Depending on if hybridization has been done or not, `F` can refer either to [mxnet.ndarray API](https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html) or [mxnet.symbol API](https://mxnet.incubator.apache.org/api/python/symbol/symbol.html). The former is used for imperative programming, and the latter for symbolic programming. 

To support hybridization, it is important to use only methods avaible directly from `F` parameter. Usually, there are equivalent methods in both APIs, but sometimes there are mismatches or small variations. For example, by default, subtraction and division of NDArrays support broadcasting, while in Symbol API broadcasting is supported in a separate operators. 

Knowing this, we can can rewrite our example layer, using HybridBlock:


```python
class NormalizationHybridLayer(gluon.HybridBlock):
    def __init__(self):
        super(NormalizationHybridLayer, self).__init__()

    def hybrid_forward(self, F, x):
        return F.broadcast_div(F.broadcast_sub(x, F.min(x)), (F.broadcast_sub(F.max(x), F.min(x))))
```

Thanks to inheriting from HybridBlock, one can easily do forward pass on a given ndarray, either on CPU or GPU:


```python
layer = NormalizationHybridLayer()
layer(nd.array([1, 2, 3], ctx=mx.cpu()))
```




    
    [0.  0.5 1. ]
    <NDArray 3 @cpu(0)>



As a rule of thumb, one should always implement custom layers by inheriting from `HybridBlock`. This allows to have more flexibility, and doesn't affect execution speed once hybridization is done. 

Unfortunately, at the moment of writing this tutorial, NLP related layers such as [RNN](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.RNN), [GRU](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.GRU) and [LSTM](https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.LSTM) are directly inhereting from the `Block` class via common `_RNNLayer` class. That means that networks with such layers cannot be hybridized. But this might change in the future, so stay tuned.

It is important to notice that hybridization has nothing to do with computation on GPU. One can train both hybridized and non-hybridized networks on both CPU and GPU, though hybridized networks would work faster. Though, it is hard to say in advance how much faster it is going to be.

## Adding a custom layer to a network

While it is possible, custom layers are rarely used separately. Most often they are used with predefined layers to create a neural network. Output of one layer is used as an input of another layer.

Depending on which class you used as a base one, you can use either [Sequential](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.nn.Sequential) or [HybridSequential](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.nn.HybridSequential) container to form a sequential neural network. By adding layers one by one, one adds dependencies of one layer's input from another layer's output. It is worth noting, that both `Sequential` and `HybridSequential` containers inherit from `Block` and `HybridBlock` respectively. 

Below is an example of how to create a simple neural network with a custom layer. In this example, `NormalizationHybridLayer` gets as an input the output from `Dense(5)` layer and pass its output as an input to `Dense(1)` layer.


```python
net = gluon.nn.HybridSequential()                         # Define a Neural Network as a sequence of hybrid blocks
with net.name_scope():                                    # Used to disambiguate saving and loading net parameters
    net.add(Dense(5))                                     # Add Dense layer with 5 neurons
    net.add(NormalizationHybridLayer())                   # Add our custom layer
    net.add(Dense(1))                                     # Add Dense layer with 1 neurons


net.initialize(mx.init.Xavier(magnitude=2.24))            # Initialize parameters of all layers
net.hybridize()                                           # Create, optimize and cache computational graph
input = nd.random_uniform(low=-10, high=10, shape=(5, 2)) # Create 5 random examples with 2 feature each in range [-10, 10]
net(input)
```




    
    [[-0.13601446]
     [ 0.26103732]
     [-0.05046433]
     [-1.2375476 ]
     [-0.15506986]]
    <NDArray 5x1 @cpu(0)>



## Parameters of a custom layer

Usually, a layer has a set of associated parameters, sometimes also referred as weights. This is an internal state of a layer. Most often, these parameters are the ones, that we want to learn during backpropogation step, but sometimes these parameters might be just constants we want to use during forward pass.

All parameters of a block are stored and accessed via [ParameterDict](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/parameter.py#L508) class. This class helps with initialization, updating, saving and loading of the parameters. Each layer can have multiple set of parameters, and all of them can be stored in a single instance of the `ParameterDict` class. On a block level, the instance of the `ParameterDict` class is accessible via `self.params` field, and outside of a block one can access all parameters of the network via [collect_params()](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.Block.collect_params) method called on a `container`. `ParameterDict` uses [Parameter](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html#mxnet.gluon.Parameter) class to represent parameters inside of Apache MxNet neural network. If parameter doesn't exist, trying to get a parameter via `self.params` will create it automatically.


```python
class NormalizationHybridLayer(gluon.HybridBlock):
    def __init__(self, hidden_units, scales):
        super(NormalizationHybridLayer, self).__init__()

        with self.name_scope():
            self.weights = self.params.get('weights',
                                           shape=(hidden_units, 0),
                                           allow_deferred_init=True)

            self.scales = self.params.get('scales',
                                      shape=scales.shape,
                                      init=mx.init.Constant(scales.asnumpy().tolist()), # Convert to regular list to make this object serializable
                                      differentiable=False)
            
    def hybrid_forward(self, F, x, weights, scales):
        normalized_data = F.broadcast_div(F.broadcast_sub(x, F.min(x)), (F.broadcast_sub(F.max(x), F.min(x))))
        weighted_data = F.FullyConnected(normalized_data, weights, num_hidden=self.weights.shape[0], no_bias=True)
        scaled_data = F.broadcast_mul(scales, weighted_data)
        return scaled_data
```

In the example above 2 set of parameters are defined:
1. Parameter `weights` is trainable. Its shape is unknown during construction phase and will be infered on the first run of forward propogation; 
1. Parameter `scale` is a constant that doesn't change. Its shape is defined during construction.

Notice a few aspects of this code:
* `name_scope()` method is used to add a prefix to parameter names during saving and loading
* Shape is not provided when creating `weights`. Instead it is going to be infered from the shape of the input
* `Scales` parameter is initialized and marked as `differentiable=False`.
* `F` backend is used for all calculations
* The calculation of dot product is done using `F.FullyConnected()` method instead of `F.dot()` method. The one was chosen over another because the former supports automatic infering shapes of inputs while the latter doesn't. This is extremely important to know, if one doesn't want to hard code all the shapes. The best way to learn what operators supports automatic inference of input shapes at the moment is browsing C++ implementation of operators to see if one uses a method `SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param.num_hidden, num_input));`
* `hybrid_forward()` method signature has changed. It accepts two new arguments: `weights` and `scales`.

The last peculiarity is due to support of imperative and symbolic programming by `HybridBlock`. During training phase, parameters are passed to the layer by Apache MxNet framework as additional arguments to the method, because they might need to be converted to a `Symbol` depending on if the layer was hybridized. One shouldn't use `self.weights` and `self.scales` or `self.params.get` in `hybrid_forward` except to get shapes of parameters. 

Running forward pass on this network is very similar to the previous example, so instead of just doing one forward pass, let's run whole training for a few epochs to show that `scales` parameter doesn't change during the training while `weights` parameter is changing.


```python
def print_params(title, net):
    """
    Helper function to print out the state of parameters of NormalizationHybridLayer
    """
    print(title)
    hybridlayer_params = {k: v for k, v in net.collect_params().items() if 'normalizationhybridlayer' in k }
    
    for key, value in hybridlayer_params.items():
        print('{} = {}\n'.format(key, value.data()))

net = gluon.nn.HybridSequential()                             # Define a Neural Network as a sequence of hybrid blocks
with net.name_scope():                                        # Used to disambiguate saving and loading net parameters
    net.add(Dense(5))                                         # Add Dense layer with 5 neurons
    net.add(NormalizationHybridLayer(hidden_units=5, 
                                     scales = nd.array([2]))) # Add our custom layer
    net.add(Dense(1))                                         # Add Dense layer with 1 neurons


net.initialize(mx.init.Xavier(magnitude=2.24))                # Initialize parameters of all layers
net.hybridize()                                               # Create, optimize and cache computational graph

input = nd.random_uniform(low=-10, high=10, shape=(5, 2))     # Create 5 random examples with 2 feature each in range [-10, 10]
label = nd.random_uniform(low=-1, high=1, shape=(5, 1))

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

    =========== Parameters after forward pass ===========
    
    hybridsequential94_normalizationhybridlayer0_weights = 
    [[-0.3983642  -0.505708   -0.02425683 -0.3133553  -0.35161012]
     [ 0.6467543   0.3918715  -0.6154656  -0.20702496 -0.4243446 ]
     [ 0.6077331   0.03922009  0.13425875  0.5729856  -0.14446527]
     [-0.3572498   0.18545026 -0.09098256  0.5106366  -0.35151464]
     [-0.39846328  0.22245121  0.13075739  0.33387476 -0.10088372]]
    <NDArray 5x5 @cpu(0)>
    
    hybridsequential94_normalizationhybridlayer0_scales = 
    [2.]
    <NDArray 1 @cpu(0)>
    
    =========== Parameters after backward pass ===========
    
    hybridsequential94_normalizationhybridlayer0_weights = 
    [[-0.29839832 -0.47213346  0.08348035 -0.2324698  -0.27368504]
     [ 0.76268613  0.43080837 -0.49052125 -0.11322092 -0.3339738 ]
     [ 0.48665082 -0.00144657  0.00376363  0.47501418 -0.23885089]
     [-0.22626656  0.22944227  0.05018325  0.6166192  -0.24941102]
     [-0.44946212  0.20532274  0.07579394  0.29261002 -0.14063817]]
    <NDArray 5x5 @cpu(0)>
    
    hybridsequential94_normalizationhybridlayer0_scales = 
    [2.]
    <NDArray 1 @cpu(0)>
    


As it is seen from the output above, `weights` parameter has been changed by the training and `scales` not.

## Conclusion

One important quality of a Deep learning framework is extensibility. Empowered by flexible abstractions, like `Block` and `HybridBlock`, one can easily extend Apache MxNet functionality to match its needs.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
