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

# Naming of Gluon Parameter and Blocks

In Gluon, each [`Parameter`](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html#mxnet.gluon.Parameter) has a name and each [`Block`](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html#mxnet.gluon.Block) has a prefix. `Parameter` names are specified by users, and `Block` names can be either specified by users or automatically created.

In this tutorial we talk about the best practices on naming. First, let's import MXNet and Gluon:

```python
from __future__ import print_function
import mxnet as mx
from mxnet import gluon
```

## Naming Blocks

When you create a `Block`, you can assign it a prefix:

```python
mydense = gluon.nn.Dense(100, prefix='mydense_')
print(mydense.prefix)
print(mydense.name)
```

The prefix of the `Block` acts like its name. As you can see from the example above, the only difference is that the last `_` is removed from the `Block`'s name.

When no prefix is given, `Gluon` will automatically generate one:

```python
dense0 = gluon.nn.Dense(100)
dense0.prefix
```

When you create more Blocks of the same kind, they will be named with an incrementing indices to avoid naming collision. To illustrate that, let's create a new `Dense` layer without specifying the prefix:

```python
dense1 = gluon.nn.Dense(100)
dense1.prefix
```

As we can see prefixes of `dense0` and `dense1` blocks are different.

## Naming Parameters

Parameters within a `Block` will be named by prepending the prefix of the `Block` to the name of the `Parameter`:

```python
dense0.collect_params()
```

As we can see, both `weight` and `bias` parameters of the `Dense` block have the same prefix `dense0_`.

If you create a new `Parameter`, for example, when you [create a custom block](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/custom_layer.html#parameters-of-a-custom-layer), you have to specify its name as Gluon cannot set a default one.

## Name scopes

To manage the names of nested Blocks, each `Block` has a `name_scope` attached to it. All Blocks created within a name scope will have its parent `Block`'s prefix prepended to its name. We can retrieve the `name_scope` by using [`.name_scope()`](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html#mxnet.gluon.Block.name_scope) method of the `Block`.

Let's demonstrate this by first defining a simple neural network:

```python
class Model(gluon.Block):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(20)
            self.dense1 = gluon.nn.Dense(20)
            self.mydense = gluon.nn.Dense(20, prefix='mydense_')

    def forward(self, x):
        x = mx.nd.relu(self.dense0(x))
        x = mx.nd.relu(self.dense1(x))
        return mx.nd.relu(self.mydense(x))
```

Here we defined three `Dense` layers, out of which only the last one has a prefix. Now, let's create and initialize our network.

```python
model0 = Model()
model0.initialize()
model0(mx.nd.zeros((1, 20)))
print(model0.prefix)
print(model0.dense0.prefix)
print(model0.dense1.prefix)
print(model0.mydense.prefix)
```

- Note that `model0.dense0` is named as `model0_dense0_` instead of `dense0_`.

- As before, to avoid naming collision, the next `Dense` layer got automatic prefix `dense1_`, and its full prefix is `model0_dense1_`.

- Also note that although we specified `mydense_` as prefix for `model.mydense`, its parent's prefix is automatically prepended to generate the prefix `model0_mydense_`.

If we create another instance of `Model`, it will be given a different name like shown before in the example with the `Dense` layer.

```python
model1 = Model()
print(model1.prefix)
print(model1.dense0.prefix)
print(model1.dense1.prefix)
print(model1.mydense.prefix)
```

Note that `model1.dense0` is still named as `dense0_` instead of `dense2_`, following `Dense` layers in previously created `model0`. This is because each instance of model's name scope is independent of each other.

**It is recommended that you manually specify a prefix for the top level `Block`, i.e. `model = Model(prefix='mymodel_')`, to avoid potential confusion in naming.**

The same principle applies to container blocks like [`Sequential`](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html#mxnet.gluon.nn.Sequential). `name_scope` can be used inside as well as outside of `__init__`. It is recommended to use `name_scope` with container blocks as the naming of the parameters inside of it tends to be more clear.

```python
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(20))
    net.add(gluon.nn.Dense(20))
print(net.prefix)
print(net[0].prefix)
print(net[1].prefix)
```

Models loaded from [`gluon.model_zoo`](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html#module-mxnet.gluon.model_zoo) behave in a similar way:

```python
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.model_zoo.vision.alexnet(pretrained=True))
    net.add(gluon.model_zoo.vision.alexnet(pretrained=True))
print(net.prefix, net[0].prefix, net[1].prefix)
```

## Saving and loading of parameters

Because `model0` and `model1` have different prefixes, their parameters also have different names:

```python
print(model0.collect_params(), '\n')
print(model1.collect_params())
```

As a result, if you try to save parameters of `model0` and then load it into `model1`, you'll get an error due to unmatching names:

```python
model0.collect_params().save('model.params')
try:
    model1.collect_params().load('model.params', mx.cpu())
except Exception as e:
    print(e)
```

To solve this problem, we use `save_parameters`/`load_parameters` instead of `collect_params` and `save`/`load`. The `save_parameters` method uses model structure instead of parameter names to match parameters.

```python
model0.save_parameters('model.params')
model1.load_parameters('model.params')
print(mx.nd.load('model.params').keys())
```

## Replacing Blocks in networks and fine-tuning

Sometimes you may want to load a pretrained model, and replace certain Blocks in it for fine-tuning. For example, the [`AlexNet`](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html#vision) model in the model zoo has 1000 output dimensions, but maybe you have only 100 classes in your application. Let's see how to change the number of output dimensions from 1000 to 100. 

The first step is to load a pretrained `AlexNet`:

```python
alexnet = gluon.model_zoo.vision.alexnet(pretrained=True)
print(alexnet.output)
print(alexnet.output.prefix)
```

- In Gluon Model Zoo, all image classification models follow the format where the feature extraction layers are named `features` while the output layer is named `output`.

- Note that the output layer is a `Dense` block with 1000 dimension outputs.

- Some networks, including `AlexNet`, provide a way to directly specify number of output classes by using `classes` argument. We will still show how to do it in a replacing block manner for demonstration purposes.

To change the output to 100 dimension, we replace it with a new `Block` which has only 100 units.

```python
with alexnet.name_scope():
    alexnet.output = gluon.nn.Dense(100)
alexnet.output.initialize()
print(alexnet.output)
print(alexnet.output.prefix)
```

## Conclusion

Each `Block` has a `prefix` and each `Parameter` has a `name`. Gluon can automatically generate unique prefixes for blocks, but it is up to you to specify Parameter's name if you create a `Parameter` yourself. Prefixes act like name spaces and create a hierarchy following the nesting of blocks in your model. To save and load model parameters, it is better to use `save_parameters`/`load_parameters` methods on the parent `Block`.

## Recommended Next Steps

- Learn more about [model serialization](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/save_load_params.html) in Gluon.

- [Create custom blocks](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/custom_layer.html) with your own parameters.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
