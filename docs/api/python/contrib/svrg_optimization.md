# SVRG Optimization in Python Module API

## Overview
SVRG which stands for Stochastic Variance Reduced Gradients, is an optimization technique that was first introduced in 
paper _Accelerating Stochastic Gradient Descent using Predictive Variance Reduction_ in 2013. It is complement to SGD 
(Stochastic Gradient Descent), which is known for large scale optimization but suffers from slow convergence 
asymptotically due to its inherent variance. SGD approximates the full gradients using a small batch of data or 
a single data sample, which will introduce variance and thus requires to start with a small learning rate in order to 
ensure convergence. SVRG remedies the problem by keeping track of a version of estimated weights that close to the 
optimal parameter values and maintaining an average of full gradients over a full pass of data. The average of full 
gradients is calculated with respect to the weights from the last m-th epochs in the training.  SVRG uses a different 
update rule: gradients w.r.t current parameter values minus gradients w.r.t to parameters from the last m-th epochs 
plus the average of full gradients over all data. 
  
Key Characteristics of SVRG:
* Employs explicit variance reduction by using a different update rule compared to SGD.
* Ability to use relatively large learning rate, which leads to faster convergence compared to SGD.
* Guarantees for fast convergence for smooth and strongly convex functions.

SVRG optimization is implemented as a SVRGModule in `mxnet.contrib.svrg_optimization`, which is an extension of the 
existing `mxnet.module.Module` APIs and encapsulates SVRG optimization logic within several new functions. SVRGModule 
API changes compared to Module API to end users are minimal. 

In distributed training, each worker gets the same special weights from the last m-th epoch and calculates the full 
gradients with respect to its own shard of data. The standard SVRG optimization requires building a global full 
gradients, which is calculated by aggregating the full gradients from each worker and averaging over the number of 
workers. The workaround is to keep an additional set of keys in the KVStore that maps to full gradients. 
The `_SVRGOptimizer` is designed to wrap two optimizers, an `_AssignmentOptimizer` which is used for full gradients 
accumulation in the KVStore and a regular optimizer that performs actual update rule to the parameters. 
The `_SVRGOptimizer` and `_AssignmentOptimizer` are designed to be used in `SVRGModule` only.

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
``` 

This document lists the SVRGModule APIs in MXNet/Contrib package:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.contrib.svrg_optimization.svrg_module
```

### Intermediate Level API for SVRGModule

The only extra step to use a SVRGModule compared to use a Module is to check if the current epoch should update the
full gradients over all data. Code snippets below demonstrate the suggested usage of SVRGModule using intermediate 
level APIs.

```python
>>> mod = SVRGModule(symbol=model, update_freq=2, data_names=['data'], label_names=['lin_reg_label'])
>>> mod.bind(data_shapes=di.provide_data, label_shapes=di.provide_label)
>>> mod.init_params()
>>> mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.01), ), kvstore='local')
>>> for epoch in range(num_epochs):
...     if epoch % mod.update_freq == 0:
...         mod.update_full_grads(di)
...     di.reset()
...     for batch in di:
...         mod.forward_backward(data_batch=batch)
...         mod.update()
```

### High Level API for SVRGModule

The high level API usage of SVRGModule remains exactly the same as Module API. Code snippets below gives an example of
suggested usage of high level API.

```python
>>> mod = SVRGModule(symbol=model, update_freq=2, data_names=['data'], label_names=['lin_reg_label'])
>>> mod.fit(di, num_epochs=100, optimizer='sgd', optimizer_params=(('learning_rate', 0.01), ))
```

## API reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.contrib.svrg_optimization.svrg_module
.. autoclass:: mxnet.contrib.svrg_optimization.svrg_module.SVRGModule
    :members: init_optimizer, bind, forward, backward, reshape, update, update_full_grads, fit, prepare
 
```
<script>auto_index("api-reference");</script>