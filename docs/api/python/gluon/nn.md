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

# Gluon Neural Network Layers

## Overview

This document lists the neural network blocks in Gluon:

```eval_rst
.. currentmodule:: mxnet.gluon.nn
```


## Basic Layers


```eval_rst
.. autosummary::
    :nosignatures:

    Dense
    Dropout
    BatchNorm
    InstanceNorm
    LayerNorm
    Embedding
    Flatten
    Lambda
    HybridLambda
```


## Convolutional Layers


```eval_rst
.. autosummary::
    :nosignatures:

    Conv1D
    Conv2D
    Conv3D
    Conv1DTranspose
    Conv2DTranspose
    Conv3DTranspose
```



## Pooling Layers


```eval_rst
.. autosummary::
    :nosignatures:

    MaxPool1D
    MaxPool2D
    MaxPool3D
    AvgPool1D
    AvgPool2D
    AvgPool3D
    GlobalMaxPool1D
    GlobalMaxPool2D
    GlobalMaxPool3D
    GlobalAvgPool1D
    GlobalAvgPool2D
    GlobalAvgPool3D
    ReflectionPad2D
```

## Activation Layers


```eval_rst
.. autosummary::
    :nosignatures:

    Activation
    LeakyReLU
    PReLU
    ELU
    SELU
    Swish
```


## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.gluon.nn
    :members:
    :imported-members:
    :exclude-members: Block, HybridBlock, SymbolBlock, Sequential, HybridSequential
```

<script>auto_index("api-reference");</script>
