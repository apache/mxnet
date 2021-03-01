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

# Parameter and Block Naming

In gluon, each Parameter or Block has a name. Parameter names and Block names can be automatically created.

In this tutorial we talk about the best practices on naming. First, let's import MXNet and Gluon:


```{.python .input}
from __future__ import print_function
import mxnet as mx
from mxnet import gluon
```

## Naming Blocks

When creating a block, you can simply do as follows:


```{.python .input}
mydense = gluon.nn.Dense(100)
print(mydense.name)
```

When you create more Blocks of the same kind, they will be named with incrementing suffixes to avoid collision:


```{.python .input}
dense1 = gluon.nn.Dense(100)
print(dense1.name)
```

## Naming Parameters

Parameters will be named automatically by a unique name in the format of `param_{uuid4}_{name}`:


```{.python .input}
param = gluon.Parameter(name = 'bias')
print(param.name)
```

`param.name` is used as the name of a parameter's symbol representation. And it can not be changed once the parameter is created.

When getting parameters within a Block, you should use the structure based name as the key:


```{.python .input}
print(dense0.collect_params())
```

## Nested Blocks

In MXNet 2, we don't have to define children blocks within a `name_scope` any more. Let's demonstrate this by defining and initiating a simple neural net:


```{.python .input}
class Model(gluon.HybridBlock):
    def __init__(self):
        super(Model, self).__init__()
        self.dense0 = gluon.nn.Dense(20)
        self.dense1 = gluon.nn.Dense(20)
        self.mydense = gluon.nn.Dense(20)

    def forward(self, x):
        x = mx.nd.relu(self.dense0(x))
        x = mx.nd.relu(self.dense1(x))
        return mx.nd.relu(self.mydense(x))

model0 = Model()
model0.initialize()
model0.hybridize()
model0(mx.nd.zeros((1, 20)))
```

The same principle also applies to container blocks like Sequential. We can simply do as follows:


```{.python .input}
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(20))
net.add(gluon.nn.Dense(20))
```


## Saving and loading


For `HybridBlock`, we use `save_parameters`/`load_parameters`, which uses model structure, instead of parameter name, to match parameters.


```{.python .input}
model0.save_parameters('model.params')
model1.load_parameters('model.params')
print(mx.nd.load('model.params').keys())
```

For `SymbolBlock.imports`, we use `export`, which uses parameter name `param.name`, to save parameters.

```{.python .input}
model0.export('model0')
model2 = gluon.SymbolBlock.imports('model0-symbol.json', ['data'], 'model0-0000.params')
```

## Replacing Blocks from networks and fine-tuning

Sometimes you may want to load a pretrained model, and replace certain Blocks in it for fine-tuning.

For example, the alexnet in model zoo has 1000 output dimensions, but maybe you only have 100 classes in your application.

To see how to do this, we first load a pretrained AlexNet.

- In Gluon model zoo, all image classification models follow the format where the feature extraction layers are named `features` while the output layer is named `output`.
- Note that the output layer is a dense block with 1000 dimension outputs.


```{.python .input}
alexnet = gluon.model_zoo.vision.alexnet(pretrained=True)
print(alexnet.output)
```


To change the output to 100 dimension, we replace it with a new block.


```{.python .input}
alexnet.output = gluon.nn.Dense(100)
alexnet.output.initialize()
```
