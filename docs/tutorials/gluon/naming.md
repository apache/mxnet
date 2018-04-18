
# Naming of Gluon Parameter and Blocks

In gluon, each Parameter or Block has a name (and prefix). Parameter names are specified by users and Block names can be either specified by users or automatically created.

In this tutorial we talk about the best practices on naming. First, let's import MXNet and Gluon:


```python
from __future__ import print_function
import mxnet as mx
from mxnet import gluon
```

## Naming Blocks

When creating a block, you can assign a prefix to it:


```python
mydense = gluon.nn.Dense(100, prefix='mydense_')
print(mydense.prefix)
```

    mydense_


When no prefix is given, Gluon will automatically generate one:


```python
dense0 = gluon.nn.Dense(100)
print(dense0.prefix)
```

    dense0_


When you create more Blocks of the same kind, they will be named with incrementing suffixes to avoid collision:


```python
dense1 = gluon.nn.Dense(100)
print(dense1.prefix)
```

    dense1_


## Naming Parameters

Parameters within a Block will be named by prepending the prefix of the Block to the name of the Parameter:


```python
print(dense0.collect_params())
```

    dense0_ (
      Parameter dense0_weight (shape=(100, 0), dtype=<type 'numpy.float32'>)
      Parameter dense0_bias (shape=(100,), dtype=<type 'numpy.float32'>)
    )


## Name scopes

To manage the names of nested Blocks, each Block has a `name_scope` attached to it. All Blocks created within a name scope will have its parent Block's prefix prepended to its name.

Let's demonstrate this by first defining a simple neural net:


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

Now let's instantiate our neural net.

- Note that `model0.dense0` is named as `model0_dense0_` instead of `dense0_`.

- Also note that although we specified `mydense_` as prefix for `model.mydense`, its parent's prefix is automatically prepended to generate the prefix `model0_mydense_`.


```python
model0 = Model()
model0.initialize()
model0(mx.nd.zeros((1, 20)))
print(model0.prefix)
print(model0.dense0.prefix)
print(model0.dense1.prefix)
print(model0.mydense.prefix)
```

    model0_
    model0_dense0_
    model0_dense1_
    model0_mydense_


If we instantiate `Model` again, it will be given a different name like shown before for `Dense`.

- Note that `model1.dense0` is still named as `dense0_` instead of `dense2_`, following dense layers in previously created `model0`. This is because each instance of model's name scope is independent of each other.


```python
model1 = Model()
print(model1.prefix)
print(model1.dense0.prefix)
print(model1.dense1.prefix)
print(model1.mydense.prefix)
```

    model1_
    model1_dense0_
    model1_dense1_
    model1_mydense_


**It is recommended that you manually specify a prefix for the top level Block, i.e. `model = Model(prefix='mymodel_')`, to avoid potential confusions in naming.**

The same principle also applies to container blocks like Sequential. `name_scope` can be used inside `__init__` as well as out side of `__init__`:


```python
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(20))
    net.add(gluon.nn.Dense(20))
print(net.prefix)
print(net[0].prefix)
print(net[1].prefix)
```

    sequential0_
    sequential0_dense0_
    sequential0_dense1_


`gluon.model_zoo` also behaves similarly:


```python
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.model_zoo.vision.alexnet(pretrained=True))
    net.add(gluon.model_zoo.vision.alexnet(pretrained=True))
print(net.prefix, net[0].prefix, net[1].prefix)
```

    sequential1_ sequential1_alexnet0_ sequential1_alexnet1_


## Saving and loading

Because model0 and model1 have different prefixes, their parameters also have different names:


```python
print(model0.collect_params(), '\n')
print(model1.collect_params())
```

    model0_ (
      Parameter model0_dense0_weight (shape=(20L, 20L), dtype=<type 'numpy.float32'>)
      Parameter model0_dense0_bias (shape=(20L,), dtype=<type 'numpy.float32'>)
      Parameter model0_dense1_weight (shape=(20L, 20L), dtype=<type 'numpy.float32'>)
      Parameter model0_dense1_bias (shape=(20L,), dtype=<type 'numpy.float32'>)
      Parameter model0_mydense_weight (shape=(20L, 20L), dtype=<type 'numpy.float32'>)
      Parameter model0_mydense_bias (shape=(20L,), dtype=<type 'numpy.float32'>)
    ) 
    
    model1_ (
      Parameter model1_dense0_weight (shape=(20, 0), dtype=<type 'numpy.float32'>)
      Parameter model1_dense0_bias (shape=(20,), dtype=<type 'numpy.float32'>)
      Parameter model1_dense1_weight (shape=(20, 0), dtype=<type 'numpy.float32'>)
      Parameter model1_dense1_bias (shape=(20,), dtype=<type 'numpy.float32'>)
      Parameter model1_mydense_weight (shape=(20, 0), dtype=<type 'numpy.float32'>)
      Parameter model1_mydense_bias (shape=(20,), dtype=<type 'numpy.float32'>)
    )


As a result, if you try to save parameters from model0 and load it with model1, you'll get an error due to unmatching names:


```python
model0.collect_params().save('model.params')
try:
    model1.collect_params().load('model.params', mx.cpu())
except Exception as e:
    print(e)
```

    Parameter 'model1_dense0_weight' is missing in file 'model.params', which contains parameters: 'model0_mydense_weight', 'model0_dense1_bias', 'model0_dense1_weight', 'model0_dense0_weight', 'model0_dense0_bias', 'model0_mydense_bias'. Please make sure source and target networks have the same prefix.


To solve this problem, we use `save_params`/`load_params` instead of `collect_params` and `save`/`load`. `save_params` uses model structure, instead of parameter name, to match parameters.


```python
model0.save_params('model.params')
model1.load_params('model.params')
print(mx.nd.load('model.params').keys())
```

    ['dense0.bias', 'mydense.bias', 'dense1.bias', 'dense1.weight', 'dense0.weight', 'mydense.weight']


## Replacing Blocks from networks and fine-tuning

Sometimes you may want to load a pretrained model, and replace certain Blocks in it for fine-tuning.

For example, the alexnet in model zoo has 1000 output dimensions, but maybe you only have 100 classes in your application.

To see how to do this, we first load a pretrained AlexNet.

- In Gluon model zoo, all image classification models follow the format where the feature extraction layers are named `features` while the output layer is named `output`.
- Note that the output layer is a dense block with 1000 dimension outputs.


```python
alexnet = gluon.model_zoo.vision.alexnet(pretrained=True)
print(alexnet.output)
print(alexnet.output.prefix)
```

    Dense(4096 -> 1000, linear)
    alexnet0_dense2_


To change the output to 100 dimension, we replace it with a new block.


```python
with alexnet.name_scope():
    alexnet.output = gluon.nn.Dense(100)
alexnet.output.initialize()
print(alexnet.output)
print(alexnet.output.prefix)
```

    Dense(None -> 100, linear)
    alexnet0_dense3_


<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
