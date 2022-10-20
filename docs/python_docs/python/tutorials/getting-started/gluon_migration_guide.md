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


# Gluon2.0: Migration Guide

## Overview
Since the introduction of the Gluon API in MXNet 1.x, it has superseded commonly used symbolic, module and model APIs for model development. In fact, Gluon was the first in the deep learning community to unify the flexibility of imperative programming with the performance benefits of symbolic programming, through just-in-time compilation.

In Gluon2.0, we extend the support to MXNet NumPy and NumPy extension with simplified interface and new functionalities:

- **Simplified hybridization with deferred compute and tracing**: Deferred compute allows the imperative execution to be used for graph construction, which allows us to unify the historic divergence of NDArray and Symbol. Hybridization now works in a simplified hybrid forward interface. Users only need to specify the computation through imperative programming. Hybridization also works through tracing, i.e. tracing the data flow of the first input data to create a graph.

- **Data 2.0**: The new design for data loading in Gluon allows hybridizing and deploying data processing pipeline in the same way as model hybridization. The new C++ data loader improves data loading efficiency on CIFAR 10 by 50%.

- **Distributed 2.0**: The new distributed-training design in Gluon 2.0 provides a unified distributed data parallel interface across native Parameter Server, BytePS, and Horovod, and is extensible for supporting custom distributed training libraries.

- **Gluon Probability**: parameterizable probability distributions and sampling functions to facilitate more areas of research such as Baysian methods and AutoML.

- **Gluon Metrics** and **Optimizers**: refactored with MXNet NumPy interface and addressed legacy issues.

Adopting these new functionalities may or may not require modifications on your models. But don't worry, this migration guide will go through a high-level mapping from old functionality to new APIs and make Gluon2.0 migration a hassle-free experience.

## Data Pipeline
**What's new**: In Gluon2.0, `MultithreadingDataLoader` is introduced to speed up the data loading pipeline. It will use the pure MXNet C++ implementation of dataloader, datasets and batchify functions. So, you can use either MXNet internal multithreading mode dataloader or python multiprocessing mode dataloader in Gluon2.0.

**Migration Guide**: Users can continue with the traditional gluon.data.Dataloader and the C++ backend will be applied automatically.

[Gluon2.0 dataloader](../../api/gluon/data/index.rst#mxnet.gluon.data.DataLoader) will provide a new parameter called `try_nopython`. This parameter takes a default value of None; when set to `True` the dataloader will compile the python dataloading pipeline into pure MXNet C++ implementation. The compilation is not guaranteed to support all use cases, but it will fallback to python in case of failure:

- The dataset is not fully [supported by the backend](../../api/gluon/data/index.rst#mxnet.gluon.data.Dataset) (e.g., there are custom python datasets).

- Transform is not fully hybridizable.

- Bachify is not fully [supported by the backend](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/data/batchify.py).


You can refer to [Step 5 in Crash Course](https://mxnet.apache.org/versions/master/api/python/docs/tutorials/getting-started/crash-course/5-datasets.html#New-in-MXNet-2.0:-faster-C++-backend-dataloaders) for a detailed performance increase with C++ backend.

## Modeling
In Gluon2.0, users will have a brand new modeling experience with NumPy-compatible APIs and the deferred compute mechanism.

- **NumPy-compatible programing experience**: users can build their models with MXNet implementation with NumPy array library, NumPy-compatible math operators and some neural network extension operators.

- **Imperative-only coding experience**: with the deferred compute and tracing being introduced, users only need to specify the computation through imperative coding but can still make hybridization work. Users will no longer need to interact with symbol APIs.

To help users migrate smoothly to use these simplified interfaces, we will provide the following guidance on how to replace legacy operators with NumPy-compatible operators, how to build models with `forward` instead of `hybrid_forward` and how to use `Parameter` class to register your parameters.


### NumPy-compatible Programming Experience
#### NumPy Arrays
MXNet [NumPy ndarray (i.e. mx.np.ndarray)](../../api/np/arrays.ndarray.rst) is a multidimensional container of items of the same type and size. Most of its properties and attributes are the same as legacy NDArrays (i.e. `mx.nd.ndarray`), so users can use the NumPy array library just as they did with legacy NDArrays. But, there are still some changes and deprecations that need attention, as mentioned below.

**Migration Guide**:

1. Currently, NumPy ndarray only supports `default` storage type, other storage types, like `row_sparse`, `csr` are not supported. Also, `tostype()` attribute is deprecated.

2. Users can use `as_np_ndarray` attribute to switch from a legacy NDArray to NumPy ndarray just like this:

```{.python}
import mxnet as mx
nd_array = mx.ones((5,3))
np_array = nd_array.as_np_ndarray()
```

3. Compared with legacy NDArray, some attributes are deprecated in NumPy ndarray. Listed below are some of the deprecated APIs and their corresponding replacements in NumPy ndarray, others can be found in [**Appendix/NumPy Array Deprecated Attributes**](#NumPy-Array-Deprecated-Attributes).

|                   Deprecated Attributes               |    NumPy ndarray Equivalent    |
| ----------------------------------------------------- | ------------------------------ |
|                   `a.asscalar()`                      |         `a.item()`             |
|                 `a.as_in_context()`                   |      `a.to_device()`           |
|                    `a.context`                        |          `a.device`            |
|                   `a.reshape_like(b)`                 |    `a.reshape(b.shape)`        |
|                    `a.zeros_like(b)`                  |   `mx.np.zeros_like(b)`        |
|                    `a.ones_like(b)`                   |   `mx.np.ones_like(b)`         |


**NOTE**

`Context` class has also been deprecated in MXNet2.0, it is renamed to `Device` and some related methods and attributes are also renamed as above. All the creation functions inside MXNet NumPy package will take `device` as keyword instead of `ctx`.


4. Compared with legacy NDArray, some attributes will have different behaviors and take different inputs. 

+--------------------------------------------------+--------------------------------------------------------------+------------------------------------------------------------------+
|                       Attribute                  |                       Legacy Inputs                          |                    NumPy Inputs                                  |
+==================================================+==============================================================+==================================================================+
|            a.reshape(*args, **kwargs)            | **shape**: Some dimensions of the shape can take special     | **shape**: shape parameter will be **positional argument** rather|
|                                                  | values from the set {0, -1, -2, -3, -4}.                     |            than key-word argument. Some dimensions of the shape  |
|                                                  | The significance of each is explained below:                 |            can take special values from the set {-1, -2, -3, -4, |
|                                                  | 0  copy this dimension from the input to the output shape.   |            -5, -6}.                                              |
|                                                  | -1 infers the dimension of the output shape by using the     | The significance of each is explained below:                     |
|                                                  |    remainder of the input dimensions.                        | -1 infers the dimension of the output shape by using the         |
|                                                  | -2 copy all/remainder of the input dimensions to the         |    remainder  of the input dimensions.                           |
|                                                  |    output shape.                                             | -2 copy this dimension from the input to the output shape.       |
|                                                  | -3 use the product of two consecutive dimensions of the      | -3 skip the current dimension if and only if the current dim size|
|                                                  |    input shape as the output dimension.                      |    is one.                                                       |
|                                                  | -4 split one dimension of the input into two dimensions      | -4 copy all the remaining the input dimensions to the output     |
|                                                  |    passed subsequent to -4 in shape (can contain -1).        |    shape.                                                        |
|                                                  | **reverse**: If set to 1, then the special values are        | -5 use the product of two consecutive dimensions of the input    |
|                                                  |              inferred from right to left                     |    shape as the output.                                          |
|                                                  |                                                              | -6 split one dimension of the input into two dimensions passed   |
|                                                  |                                                              |    subsequent to -6 in the new shape.                            |
|                                                  |                                                              | **reverse**: No **reverse** parameter for `np.reshape` but for   |
|                                                  |                                                              |              `npx.reshape`.                                      |
|                                                  |                                                              | **order**: Read the elements of `a` using this index order, and  |
|                                                  |                                                              |            place the elements into the reshaped array using this |
|                                                  |                                                              |            index order.                                          |
+--------------------------------------------------+--------------------------------------------------------------+------------------------------------------------------------------+



#### NumPy and NumPy-extension Operators
Most of the legacy NDArray operators (`mx.nd.op`) have the equivalent ones in np/npx namespace. Users can just replace them with `mx.np.op` or `mx.npx.op` to migrate. Some of the operators will have different inputs and behaviors as listed in the table below.

**Migration Guide**:

1. Operators migration with name/inputs changes

+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|                   Legacy Operators               |               NumPy Operators Equivalent                |                              Changes                                  |
+==================================================+=========================================================+=======================================================================+
|        mx.nd.flatten(*args, **kwargs)            |        mx.npx.batch_flatten(*args, **kwargs)            |     moved to npx namespace with new name batch_flatten                |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|       mx.nd.concat(a, b, c)                      |            mx.np.concatenate([a, b, c])                 |       - moved to np namespace with new name concatenate.              |
|                                                  |                                                         |       - use list of ndarrays as input rather than positional ndarrays |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|        mx.nd.stack(a, b, c)                      |            mx.np.stack([a, b, c])                       |       - moved to np namespace.                                        |
|                                                  |                                                         |       - use list of ndarrays as input rather than positional ndarrays |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|      mx.nd.SliceChannel(*args, **kwargs)         |            mx.npx.slice_channel(*args, **kwargs)        |         moved to npx namespace with new name slice_channel.           |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|      mx.nd.FullyConnected(*args, **kwargs)       |        mx.npx.fully_connected(*args, **kwargs)          |         moved to npx namespace with new name fully_connected.         |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|      mx.nd.Activation(*args, **kwargs)           |            mx.npx.activation(*args, **kwargs)           |         moved to npx namespace with new name activation.              |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|      mx.nd.elemwise_add(a, b)                    |            a + b                                        |         Just use ndarray python operator.                             |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|      mx.nd.elemwise_mul(a, b)                    |            mx.np.multiply(a, b)                         |              Use multiply operator in np namespace.                   |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+

2. Operators migration with multiple steps: `mx.nd.mean` -> `mx.np.mean`:

```{.python}
import mxnet as mx
# Legacy: calculate mean value with reduction on axis 1
#         with `exclude` option on 
nd_mean = mx.nd.mean(data, axis=1, exclude=1)

# Numpy: no exclude option to users, but user can perform steps as follow
axes = list(range(data.ndim))
del axes[1]
np_mean = mx.np.mean(data, axis=axes)
```

3. Random Operators

+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|                   Legacy Operators               |               NumPy Operators Equivalent                |                              Changes                                  |
+==================================================+=========================================================+=======================================================================+
|   `mx.random.uniform(-1.0, 1.0, shape=(2, 3))`   |       `mx.np.random.uniform(-1.0, 1.0, size=(2, 3))`    |   For all the NumPy random operators, use **size** keyword instead of |
|  `mx.nd.random.uniform(-1.0, 1.0, shape=(2, 3))` |                                                         |   **shape**                                                           |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|    `mx.nd.random.multinomial(*args, **kwargs)`   |       `mx.npx.random.categorical(*args, **kwargs)`      |   use `npx.random.categorical` to have the behavior of drawing 1 sample from multiple distributions.  |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+

4. Control Flow Operators

+----------------------------------------------------------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
|                               Legacy Operators                             |                NumPy Operators Equivalent                                 |                             Changes                                                                           |
+============================================================================+===========================================================================+===============================================================================================================+
|          `mx.nd.contrib.foreach(body, data, init_states, name)`            |    `mx.npx.foreach(body, data, init_states, name)`                        | - moved to `npx` namespace.                                                                        |
|                                                                            |                                                                           | - Will not support global variables as body's inputs(body's inputs must be either data or states or both)   |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
|  `mx.nd.contrib.while_loop(cond, func, loop_vars, max_iterations, name)`   |    `mx.npx.while_loop(cond, func, loop_vars, max_iterations, name)`       | - moved to `npx` namespace.                                                                        |
|                                                                            |                                                                           | - Will not support global variables as cond or func's inputs(cond or func's inputs must be in loop_vars)    |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+
|       `mx.nd.contrib.cond(pred, then_func, else_func, inputs, name)`       |        `mx.npx.cond(pred, then_func, else_func, name)`                    | - moved to `npx` namespace.                                                                        |
|                                                                            |                                                                           | - users needs to provide the inputs of pred, then_func and else_func as inputs                             |
|                                                                            |                                                                           | - Will not support global variables as pred, then_func or else_func's                                       |
|                                                                            |                                                                           | inputs(pred, then_func or else_func's inputs must be in inputs)                                             |
+----------------------------------------------------------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------+

5. Functionalities

+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|                   Legacy Operators               |               NumPy Operators Equivalent                |                              Changes                                  |
+==================================================+=========================================================+=======================================================================+
|       `mx.nd.save(*args, **kwargs)`              |            `mx.npx.savez(*args, **kwargs)`              |  - moved to `npx` namespace.                                          |
|                                                  |                                                         |  - Only accept positional arguments, try to flatten the list/dict     |
|                                                  |                                                         |    before feed in                                                     |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|       `mx.nd.load(*args, **kwargs)`              |            `mx.npx.load(*args, **kwargs)`               |  - moved to `npx` namespace.                                          |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+
|       `mx.nd.waitall()`                          |            `mx.npx.waitall()`                           |  - moved to `npx` namespace.                                          |
+--------------------------------------------------+---------------------------------------------------------+-----------------------------------------------------------------------+

Other operator changes are included in [**Appendix/NumPy and NumPy-extension Operators**](#NumPy-and-NumPy-extension-Operators1) 



### Layers and Blocks
With the deferred compute and tracing being introduced in Gluon2.0, users do not need to interact with symbols any more. There are a lot of changes in building a model with Gluon API, including parameter management and naming, forward pass computing and parameter shape inferencing. We provide step-by-step migration guidance on how to build a model with new APIs. 

#### Parameter Management and Block Naming
In Gluon, each Parameter or Block has a name (and prefix). Parameter names are specified by users and Block names can be either specified by users or automatically created. In Gluon 1.x, parameters are accessed via the `params` variable of the `ParameterDict` in `Block`. Users will need to manually use `with self.name_scope():` for children blocks and specify prefix for the top level block. Otherwise, it will lead to wrong name scopes and can return parameters of children blocks that are not in the current name scope. An example for initializing the Block and Parameter in Gluon 1.x: 
```{.python}
from mxnet.gluon import Parameter, Constant, HybridBlock
class SampleBlock(HybridBlock):
    def __init__(self):
        super(SampleBlock, self).__init__()
        with self.name_scope():
            # Access parameters, which are iterated during training
            self.weight = self.params.get('weight')
            # Access constant parameters, which are not iterated during training
            self.weight = self.params.get_constant('const', const_arr)
```
Now in Gluon 2.0, Block/HybridBlock objects will not maintain the parameter dictionary (`ParameterDict`). Instead, users can access these parameters via `Parameter` class and `Constant` class. These parameters will be registered automatically as part of the Block. Users will no longer need to manage the name scope for children blocks and hence can remove `with self.name_scope():` this statement. For example:
```{.python}
class SampleBlock(HybridBlock):
    def __init__(self):
        super(SampleBlock, self).__init__()
        # Access parameters, which are iterated during training
        self.weight = Parameter('weight')
        # Access constant parameters, which are not iterated during training
        self.weight = Constant('const', const_arr)
```
Also, there will be new mechanisms for parameter loading, sharing and setting device. 

1. Parameter loading in Gluon 1.x vs Gluon 2.0:

```{.python}
# in Gluon 1.x
net = nn.Dense(8, activation='relu')
net.collect_params().load_dict(arg_dict, ctx=ctx)
# in Gluon 2.0
net = nn.Dense(8, activation='relu')
net.load_dict(arg_dict, device=device)
```

2. Parameter sharing in Gluon 1.x vs Gluon 2.0:

```{.python}
# in Gluon 1.x
shared = nn.Dense(8, activation='relu')
net = nn.Dense(8, activation='relu', params=shared.params)
# in Gluon 2.0
shared = nn.Dense(8, activation='relu')
net = nn.Dense(8, activation='relu').share_parameters(shared.params)
```

3. Parameter setting device in Gluon 1.x vs Gluon 2.0:

```{.python}
# in Gluon 1.x
net = nn.Dense(8, activation='relu')
net.collect_params().reset_ctx(devices)
# in Gluon 2.0
net = nn.Dense(8, activation='relu')
net.reset_device(devices)
```

#### Forward Interface
`hybrid_forward` interface in Gluon1.x provides the user with a unified imperative and symbolic programming interface to do graph construction and imperative execution. For the inputs of `hybrid_forward`, `F` can be either mx.symbol or mx.ndarray depending on the running mode(symbolic or imperative) of variable recording. Apart from `F` and input arrays, the parameters registered when Block is initialized are also required as part of the inputs. Take `nn.Dense` as an example:

```{.python}
# hybrid_forward interface, F can be either symbol or ndarray, weights
# and bias are part of inputs
def hybrid_forward(self, F, x, weight, bias=None):
    fc = F.npx.fully_connected if is_np_array() else F.FullyConnected
    act = fc(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
             flatten=self._flatten, name='fwd')
    if self.act is not None:
        act = self.act(act)
    return act
```

Now, in deferred computation mode of Gluon2.0, the divergence of NDArray and Symbol is unified, which means users no longer need to define `F` with specific running mode. One can easily specify the computation through imperative programming, hybridization will work through the tracing mechanism(data flow of the first input batch). What's more, users can implement the forward interface with `npx/npx` operators instead of `nd` and `symbol`. 

```{.python}
# forward interface, no F any more
def forward(self, x):
    # get the device information of input array and make parameters run on the same device
    device = x.device
    # use np/npx interfaces instead of F
    act = npx.fully_connected(x, self.weight.data(device),
                              self.bias.data(device) if self.bias is not None else None,
                              no_bias=self.bias is None,
                              num_hidden=self._units, flatten=self._flatten, name='fwd')
    if self.act is not None:
        act = self.act(act)
    return act
```

#### Implement Infer Shape
In Gluon1.x, parameter shape inference happens in MXNet backend. Now in Gluon2.0, shape inference is disabled in the case of deferred parameter initialization. So, users should now always implement `infer_shape` method to set the parameter shapes if the parameter shape was not set during HybridBlock initialization. 

```{.python}
def infer_shape(self, x, *args):
    # if true, self.weight.shape[1] will be flattened of input's shape
    if self._flatten:
        num_input = 1
        for i in range(1, x.ndim):
            num_input *= x.shape[i]
        self.weight.shape = (self.weight.shape[0], num_input)
    # if false, self.weight.shape[1] = x.shape[-1]
    else:
        self.weight.shape = (self.weight.shape[0], x.shape[x.ndim - 1])
```

Now, in Gluon2.0, users can implement a Dense Block like this: 

```{.python}
class Dense(HybridBlock):
    def __init__(self, units, activation=None, use_bias=True, flatten=True,
                 dtype='float32', weight_initializer=None, bias_initializer='zeros',
                 in_units=0, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self._flatten = flatten
        self._units = units
        self._in_units = in_units
        self.weight = Parameter('weight', shape=(units, in_units),
                                init=weight_initializer, dtype=dtype,
                                allow_deferred_init=True)
        if use_bias:
            self.bias = Parameter('bias', shape=(units,),
                                  init=bias_initializer, dtype=dtype,
                                  allow_deferred_init=True)
        else:
            self.bias = None
        if activation is not None:
            self.act = Activation(activation)
        else:
            self.act = None

    def forward(self, x):
        device = x.device
        act = npx.fully_connected(x, self.weight.data(device),
                                  self.bias.data(device) if self.bias is not None else None,
                                  no_bias=self.bias is None,
                                  num_hidden=self._units, flatten=self._flatten, name='fwd')
        if self.act is not None:
            act = self.act(act)
        return act

    def infer_shape(self, x, *args):
        if self._flatten:
            num_input = 1
            for i in range(1, x.ndim):
                num_input *= x.shape[i]
            self.weight.shape = (self.weight.shape[0], num_input)
        else:
            self.weight.shape = (self.weight.shape[0], x.shape[x.ndim - 1])
```

## Optimizers
Optimizer module in MXNet provides a lot of optimization algorithms to reduce the training error. In Gluon 2.0, optimizers will also switch to use MXNet NumPy-compatible interface. Some important changes that needs attention are: 

1. AdaGrad: 
    - use `epsilon` instead of `eps`
    - e.g. `adagrad_optimizer = optimizer.AdaGrad(learning_rate=0.1, epsilon=1e-07)`

2. RMSProp:
    - use `rho` instead of `gamma1` and use `momentum` instead of `gamma2`
    - e.g. `rmsprop_optimizer = optimizer.RMSProp(learning_rate=0.001, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False)`

3. `optimizer.ccSGD` and `optimizer.LBSGD` are deprecated.

## Metrics
Metrics module in MXNet provides different methods for users to judge the performance of models. In Gluon 2.0, metrics will use MXNet NumPy-compatible interface and also introduce a lot of new evaluation metrics.
**Changes**:
1. metric module has been moved to gluon namespace
    - `mxnet.metric` -> `mxnet.gluon.metric`

2. Add new evaluation metrics: 
    - `Class BinaryAccuracy(threshold=0.5)`
    - `Class MeanCosineSimilarity(axis=-1, eps=1e-12)`
    - `Class MeanPairwiseDistance(p=2)`
    - `Class Fbeta(class_type="binary", beta=1, threshold=0.5, average="micro")`

3. Improve Class F1
    - `Class F1(name='f1',output_names=None, label_names=None, average="macro")` to
      `Class F1(name='f1',output_names=None, label_names=None, class_type="binary", threshold=0.5, average="micro")`
    - **average**: Strategy to be used for aggregating across mini-batches.
        - "macro": Calculate metrics for each label and return unweighted mean of f1.
        - "micro": Calculate metrics globally by counting the total TP, FN and FP.
        - None: Return f1 scores for each class (numpy.ndarray).
    - **class_type**:
        - "binary": f1 for binary classification.
        - "multiclass": f1 for multiclassification problem.
        - "multilabel": f1 for multilabel classification.
    - **threshold**: threshold for postive confidence value.


## Key-Value Store
Gluon 2.0 will provide a new and unified low level API for data parallel training. These unified APIs can support different communication backends, including native Parameter Server, Horovod and BytePS. 
Example: 

```{.python}
import mxnet as mx
# create key-value store with horovod backend
kv = mx.kv.create('horovod') # or choose 'kvstore', 'byteps' as backend
device = mx.gpu(kv.local_rank) if mx.device.num_gpus() > 0 else mx.cpu(kv.local_rank)
val = mx.np.zeros((2, 3), device=device)
# broadcast the value at rank 0 to all ranks
kv.broadcast('0', mx.np.zeros((2, 3), device=device), out=val)
scale = kv.rank + 1
# performs allreduce on a single array
kv.pushpull('3', val * scale)
```

## Probability
A new module called `mxnet.gluon.probability` has been introduced in Gluon 2.0. It is analogous to pytorch distribution and the main difference is that `mxnet.gluon.probability` will use MXNet NumPy compatible operators and will allow hybridization. It has three parts: 

1. [Distribution Objects](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon/probability/distributions): `gluon.probability.Bernoulli`, `gluon.probability.Beta` ...

2. [StochasticBlock](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon/probability/block): support accumulating loss in the forward phase, which is useful in building Bayesian Neural Network. 

3. [Transformation](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon/probability/transformation): implement invertible transformation with computable log det jacobians.

##  oneDNN Integration
### Operator Fusion
In versions 1.x of MXNet pattern fusion in execution graph was enabled by default when using MXNet built with oneDNN library support and could have been disabled by setting 'MXNET_SUBGRAPH_BACKEND' environment flag to `None`. MXNet 2.0 introduced changes in forward inference flow which led to refactor of fusion mechanism. To fuse model in MXNet 2.0 there are two requirements:

 - the model must be defined as a subclass of HybridBlock or Symbol,

 - the model must have specific operator patterns which can be fused.

Both HybridBlock and Symbol classes provide API to easily run fusion of operators. Adding only one line of code is needed to run fusion passes on model:
```{.python}
# on HybridBlock
net.optimize_for(data, backend='ONEDNN')
# on Symbol
optimized_symbol = sym.optimize_for(backend='ONEDNN')
```

Controling which patterns should be fused still can be done by setting proper environment variables. See [**oneDNN Environment Variables**](#oneDNN-Environment-Variables)

### INT8 Quantization / Precision reduction
Quantization API was also refactored to be consistent with other new features and mechanisms. In comparison to MXNet 1.x releases, in MXNet 2.0 `quantize_net_v2` function has been removed and development focused mainly on `quantize_net` function to make it easier to use for end user and ultimately give him more flexibility.
Quantization can be performed on either subclass of HybridBlock with `quantize_net` or Symbol with deprecated `quantize_model` (`quantize_model` is left only to provide backward compatibility and its usage is strongly discouraged).

```{.python}
import mxnet as mx
from mxnet.contrib.quantization import quantize_net
from mxnet.gluon.model_zoo.vision import resnet50_v1

# load model
net = resnet50_v1(pretrained=True)

# prepare calibration data
dummy_data = mx.nd.random.uniform(-1.0, 1.0, (batch_size, 3, 224, 224))
calib_data_loader = mx.gluon.data.DataLoader(dummy_data, batch_size=batch_size)

# quantization
qnet = quantize_net(net, calib_mode='naive', calib_data=calib_data_loader)
```
`quantize_net` can be much more complex - all function attributes can be found in the [API](../../api/contrib/quantization/index.rst).

### oneDNN Environment Variables
In version 2.0 of MXNet all references to MKLDNN (former name of oneDNN) were replaced by ONEDNN. Below table lists all environment variables:

|               MXNet 1.x              |                MXNet 2.0               |
| ------------------------------------ | ---------------------------------------|
|         MXNET_MKLDNN_ENABLED         |          MXNET_ONEDNN_ENABLED          |
|         MXNET_MKLDNN_CACHE_NUM       |         MXNET_ONEDNN_CACHE_NUM         |
|    MXNET_MKLDNN_FORCE_FC_AB_FORMAT   |     MXNET_ONEDNN_FORCE_FC_AB_FORMAT    |
|         MXNET_MKLDNN_ENABLED         |          MXNET_ONEDNN_ENABLED          |
|         MXNET_MKLDNN_DEBUG           |           MXNET_ONEDNN_DEBUG           |
|         MXNET_USE_MKLDNN_RNN         |          MXNET_USE_ONEDNN_RNN          |
|     MXNET_DISABLE_MKLDNN_CONV_OPT    |      MXNET_DISABLE_ONEDNN_CONV_OPT     |
|    MXNET_DISABLE_MKLDNN_FUSE_CONV_BN |    MXNET_DISABLE_ONEDNN_FUSE_CONV_BN   |
|  MXNET_DISABLE_MKLDNN_FUSE_CONV_RELU |   MXNET_DISABLE_ONEDNN_FUSE_CONV_RELU  |
|  MXNET_DISABLE_MKLDNN_FUSE_CONV_SUM  |   MXNET_DISABLE_ONEDNN_FUSE_CONV_SUM   |
|      MXNET_DISABLE_MKLDNN_FC_OPT     |       MXNET_DISABLE_ONEDNN_FC_OPT      |
| MXNET_DISABLE_MKLDNN_FUSE_FC_ELTWISE |  MXNET_DISABLE_ONEDNN_FUSE_FC_ELTWISE  |
| MXNET_DISABLE_MKLDNN_TRANSFORMER_OPT |  MXNET_DISABLE_ONEDNN_TRANSFORMER_OPT  |
|                  n/a                 |   MXNET_DISABLE_ONEDNN_BATCH_DOT_FUSE  |
|                  n/a                 |      MXNET_ONEDNN_FUSE_REQUANTIZE      |
|                  n/a                 |      MXNET_ONEDNN_FUSE_DEQUANTIZE      |

## Appendix
### NumPy Array Deprecated Attributes
|                   Deprecated Attributes               |    NumPy ndarray Equivalent    |
| ----------------------------------------------------- | ------------------------------ |
|                   `a.abs()`                           |             `mx.np.abs(a)`           |
|                   `a.sign()`                          |             `mx.np.sign(a)`          |
|              `a.split_v2(2, axis=1)`                  |   `mx.np.split(a, 2, axis=1)`  |
|            `a.flip(*args, **kwargs)`                  |    `mx.np.flip(a, *args, **kwargs)`  |
|            `a.diag(*args, **kwargs)`                  |    `mx.np.diag(a, *args, **kwargs)`  |
|           `a.nansum(*args, **kwargs)`                 | `mx.np.nan_to_num(a, *args, **kwargs).sum()`  |
|           `a.nanprod(*args, **kwargs)`                | `mx.np.nan_to_num(a, *args, **kwargs).prod()` |
|            `a.diag(*args, **kwargs)`                  |    `mx.np.diag(a, *args, **kwargs)`  |
|                  `a.norm()`                           |           `mx.npx.norm(a)`           |
|            `a.rint(*args, **kwargs)`                  |    `mx.np.rint(a, *args, **kwargs)`  |
|            `a.fix(*args, **kwargs)`                   |    `mx.np.fix(a, *args, **kwargs)`   |
|            `a.floor(*args, **kwargs)`                 |    `mx.np.floor(a, *args, **kwargs)`  |
|            `a.ceil(*args, **kwargs)`                  |    `mx.np.ceil(a, *args, **kwargs)`   |
|            `a.trunc(*args, **kwargs)`                 |    `mx.np.trunc(a, *args, **kwargs)`  |
|            `a.sin(*args, **kwargs)`                   |    `mx.np.sin(a, *args, **kwargs)`    |
|            `a.cos(*args, **kwargs)`                   |    `mx.np.cos(a, *args, **kwargs)`    |
|            `a.tan(*args, **kwargs)`                   |    `mx.np.tan(a, *args, **kwargs)`    |
|            `a.arcsin(*args, **kwargs)`                |    `mx.np.arcsin(a, *args, **kwargs)`  |
|            `a.arccos(*args, **kwargs)`                |    `mx.np.arccos(a, *args, **kwargs)`  |
|            `a.arctan(*args, **kwargs)`                |    `mx.np.arctan(a, *args, **kwargs)`  |
|            `a.degrees(*args, **kwargs)`               |    `mx.np.degrees(a, *args, **kwargs)`  |
|            `a.radians(*args, **kwargs)`               |    `mx.np.radians(a, *args, **kwargs)`  |
|            `a.sinh(*args, **kwargs)`                  |    `mx.np.sinh(a, *args, **kwargs)`  |
|            `a.cosh(*args, **kwargs)`                  |    `mx.np.cosh(a, *args, **kwargs)`  |
|            `a.tanh(*args, **kwargs)`                  |    `mx.np.tanh(a, *args, **kwargs)`  |
|            `a.arcsinh(*args, **kwargs)`               |    `mx.np.arcsinh(a, *args, **kwargs)`  |
|            `a.arccosh(*args, **kwargs)`               |    `mx.np.arccosh(a, *args, **kwargs)`  |
|            `a.arctanh(*args, **kwargs)`               |    `mx.np.arctanh(a, *args, **kwargs)`  |
|            `a.exp(*args, **kwargs)`                   |    `mx.np.exp(a, *args, **kwargs)`  |
|            `a.expm1(*args, **kwargs)`                 |    `mx.np.expm1(a, *args, **kwargs)`  |
|            `a.log(*args, **kwargs)`                   |    `mx.np.log(a, *args, **kwargs)`  |
|            `a.log10(*args, **kwargs)`                 |    `mx.np.log10(a, *args, **kwargs)`  |
|            `a.log2(*args, **kwargs)`                  |    `mx.np.log2(a, *args, **kwargs)`  |
|            `a.log1p(*args, **kwargs)`                 |    `mx.np.log1p(a, *args, **kwargs)`  |
|            `a.sqrt(*args, **kwargs)`                  |    `mx.np.sqrt(a, *args, **kwargs)`  |
|            `a.rsqrt(*args, **kwargs)`                 |    `1 / mx.np.sqrt(a, *args, **kwargs)`  |
|            `a.cbrt(*args, **kwargs)`                  |    `mx.np.cbrt(a, *args, **kwargs)`  |
|            `a.rcbrt(*args, **kwargs)`                 |    `1 / mx.np.cbrt(a, *args, **kwargs)`  |
|            `a.square(*args, **kwargs)`                |    `mx.np.square(a, *args, **kwargs)`  |
|                `a.pad(*args, **kwargs)`               |   `mx.npx.pad(a, *args, **kwargs)`   |
|          `a.split(axis=1, num_outputs=2)`             |   `mx.np.split(a, 2, axis=1)`  |
|            `a.slice(*args, **kwargs)`                 |   `mx.npx.slice(a, *args, **kwargs)`  |
|          `a.one_hot(*args, **kwargs)`                 |   `mx.npx.one_hot(a, *args, **kwargs)`  |
|           `a.pick(*args, **kwargs)`                   |   `mx.npx.pick(a, *args, **kwargs)`  |
|           `a.topk(*args, **kwargs)`                   |   `mx.npx.topk(a, *args, **kwargs)`  |
|               `a.shape_array()`                       |         `mx.np.array(a.shape)`       |
|               `a.size_array()`                        |         `mx.np.array(a.size)`        |
|         `a.expand_dims(*args, **kwargs)`              | `mx.np.expand_dims(a, *args, **kwargs)`  |
|            `a.relu(*args, **kwargs)`                  |    `mx.npx.relu(a, *args, **kwargs)`  |
|            `a.sigmoid(*args, **kwargs)`               |    `mx.npx.sigmoid(a, *args, **kwargs)`  |
|            `a.softmax(*args, **kwargs)`               |    `mx.npx.softmax(a, *args, **kwargs)`  |
|            `a.log_softmax(*args, **kwargs)`           |    `mx.npx.log_softmax(a, *args, **kwargs)`  |
|        `a.broadcast_like(*args, **kwargs)`            |  `mx.npx.broadcast_like(a, *args, **kwargs)`  |
|            `a.reciprocal(*args, **kwargs)`            |    `mx.np.reciprocal(a, *args, **kwargs)`  |

### NumPy and NumPy-extension Operators
|                   Legacy Operators               |    NumPy Operators Equivalent    |   Changes  |
| ----------------------------------------------------- | ------------------------------ | ------------------- |
|       `mx.nd.softmax(*args, **kwargs)`                |            `mx.npx.softmax(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.log_softmax(*args, **kwargs)`                |            `mx.npx.log_softmax(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.masked_softmax(*args, **kwargs)`                |            `mx.npx.masked_softmax(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.masked_log_softmax(*args, **kwargs)`                |            `mx.npx.masked_log_softmax(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.pick(*args, **kwargs)`                |            `mx.npx.pick(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.topk(*args, **kwargs)`                |            `mx.npx.topk(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.batch_dot(*args, **kwargs)`                |            `mx.npx.batch_dot(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.broadcast_like(*args, **kwargs)`                |            `mx.npx.broadcast_like(*args, **kwargs)`                    |                moved to `npx` namespace            |
|       `mx.nd.arange_like(*args, **kwargs)`                |            `mx.npx.arange_like(*args, **kwargs)`                    |                moved to `npx` namespace            |
|      `mx.nd.BatchNorm(*args, **kwargs)`              |            `mx.npx.batch_norm(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `batch_norm`.          |
|      `mx.nd.Convolution(*args, **kwargs)`              |            `mx.npx.convolution(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `convolution`.          |
|      `mx.nd.Deconvolution(*args, **kwargs)`              |            `mx.npx.deconvolution(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `deconvolution`.          |
|      `mx.nd.Pooling(*args, **kwargs)`              |            `mx.npx.pooling(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `pooling`.          |
|      `mx.nd.Dropout(*args, **kwargs)`              |            `mx.npx.dropout(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `dropout`.          |
|      `mx.nd.RNN(*args, **kwargs)`              |            `mx.npx.rnn(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `rnn`.          |
|      `mx.nd.Embedding(*args, **kwargs)`              |            `mx.npx.embedding(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `embedding`.          |
|      `mx.nd.LayerNorm(*args, **kwargs)`              |            `mx.npx.layer_norm(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `layer_norm`.          |
|      `mx.nd.LeakyReLU(*args, **kwargs)`              |            `mx.npx.leaky_relu(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `leaky_relu`.          |
|      `mx.nd.GroupNorm(*args, **kwargs)`              |            `mx.npx.group_norm(*args, **kwargs)`                 |              - moved to `npx` namespace with new name `group_norm`.          |

## Reference

1. [Next Generation of GluonNLP](https://github.com/dmlc/gluon-nlp/tree/master)
2. [MXNet NumPy-compatible coding experience](https://github.com/apache/incubator-mxnet/issues/14253)
3. [Gluon Data API Extension](https://github.com/apache/incubator-mxnet/issues/17269)
4. [Simplifying MXNet Gluon APIs](https://github.com/apache/incubator-mxnet/issues/18412)
5. [Deferred Compute and Tracing](https://github.com/apache/incubator-mxnet/issues/16376)
6. [MXNet Metrics Improvements](https://github.com/apache/incubator-mxnet/issues/18046)
7. [Gluon Distribution Module](https://github.com/apache/incubator-mxnet/issues/17240)