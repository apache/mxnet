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
# Step 2: Create a neural network  
  
In this step, you learn how to use NP on Apache MXNet to create neural networks  
in Gluon. In addition to the `np` package that you learned about in the previous  
step [Step 1: Manipulate data with NP on MXNet](1-nparray.md), you also need to  
import the neural network modules from `gluon`. Gluon includes built-in neural  
network layers in the following two modules:  
  
1. `mxnet.gluon.nn`: NN module that maintained by the mxnet team  
2. `mxnet.gluon.contrib.nn`: Experiemental module that is contributed by the  
community  
  
Use the following commands to import the packages required for this step.  
  
```python  
from mxnet import np, npx  
from mxnet.gluon import nn  
npx.set_np()  # Change MXNet to the numpy-like mode.  
```  
  
## Create your neural network's first layer  
  
In this section, you will create a simple neural network with Gluon. One of the  
simplest network you can create is a single **Dense** layer or **densely-  
connected** layer. A dense layer consists of nodes in the input that are  
connected to every node in the next layer. Use the following code example to  
start with a dense layer with five output units.  
  
```python  
layer = nn.Dense(5)  
layer   
# output: Dense(-1 -> 5, linear)  
```  
  
In the example above, the output is `Dense(-1 -> 5, linear)`. The **-1** in the  
output denotes that the size of the input layer is not specified during  
initialization.  
  
You can also call the **Dense** layer with an `in_units` parameter if you know  
the shape of your input unit.  
  
```python  
layer = nn.Dense(5,in_units=3)  
layer  
```  
  
In addition to the `in_units` param, you can also add an activation function to  
the layer using the `activation` param. The Dense layer implements the operation  
  
$$output = \sigma(W \cdot X + b)$$  
  
Call the Dense layer with an `activation` parameter to use an activation  
function.  
  
```python  
layer = nn.Dense(5, in_units=3,activation='relu')  
```  
  
Voila! Congratulations on creating a simple neural network. But for most of your  
use cases, you will need to create a neural network with more than one dense  
layer or with multiple types of other layers. In addition to the `Dense` layer,  
you can find more layers at [mxnet nn  
layers](https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html#module-  
mxnet.gluon.nn)  
  
So now that you have created a neural network, you are probably wondering how to  
pass data into your network?  
  
First, you need to initialize the network weights, if you use the default  
initialization method which draws random values uniformly in the range $[-0.7,  
0.7]$. You can see this in the following example.  
  
**Note**: Initialization is discussed at a little deeper detail in the next  
notebook  
  
```python  
layer.initialize()  
```  
  
Now that you have initialized your network, you can give it data. Passing data  
through a network is also called a forward pass. You can do a forward pass with  
random data, shown in the following example. First, you create a `(10,3)` shape  
random input `x` and feed the data into the layer to compute the output.  
  
```python  
x = np.random.uniform(-1,1,(10,3))  
layer(x)  
```  
  
The layer produces a `(10,5)` shape output from your `(10,3)` input.  
  
**When you don't specify the `in_unit` parameter, the system  automatically  
infers it during the first time you feed in data during the first forward step  
after you create and initialize the weights.**  
  
  
```python  
layer.params  
```  
  
The `weights` and `bias` can be accessed using the `.data()` method.  
  
```python  
layer.weight.data()  
```  
  
## Chain layers into a neural network using nn.Sequential  
  
Sequential provides a special way of rapidly building networks when when the  
network architecture follows a common design pattern: the layers look like a  
stack of pancakes. Many networks follow this pattern: a bunch of layers, one  
stacked on top of another, where the output of each layer is fed directly to the  
input to the next layer. To use sequential, simply provide a list of layers  
(pass in the layers by calling `net.add(<Layer goes here!>`). To do this you can  
use your previous example of Dense layers and create a 3-layer multi layer  
perceptron. You can create a sequential block using `nn.Sequential()` method and  
add layers using `add()` method.  
  
```python  
net = nn.Sequential()  
  
net.add(nn.Dense(5,in_units=3,activation='relu'),  
 nn.Dense(25, activation='relu'), nn.Dense(2) )  
net  
```  
  
The layers are ordered exactly the way you defined your neural network with  
index starting from 0. You can access the layers by indexing the network using  
`[]`.  
  
```python  
net[1]  
```  
  
## Create a custom neural network architecture flexibly  
  
`nn.Sequential()` allows you to create your multi-layer neural network with  
existing layers from `gluon.nn`. It also includes a pre-defined `forward()`  
function that sequentially executes added layers. But what if the built-in  
layers are not sufficient for your needs. If you want to create networks like  
ResNet which has complex but repeatable components, how do you create such a  
network?  
  
In gluon, every neural network layer is defined by using a base class  
`nn.Block()`. A Block has one main job - define a forward method that takes some  
input x and generates an output. A Block can just do something simple like apply  
an activation function. It can combine multiple layers together in a single  
block or also combine a bunch of other Blocks together in creative ways to  
create complex networks like Resnet. In this case, you will construct three  
Dense layers. The `forward()` method can then invoke the layers in turn to  
generate its output.  
  
Create a subclass of `nn.Block` and implement two methods by using the following  
code.  
  
- `__init__` create the layers  
- `forward` define the forward function.  
  
```  
class Net(nn.Block):  
 def __init__(self): super().__init__()  
 def forward(self, x): return x```  
  
```python  
class MLP(nn.Block):  
 def __init__(self): super().__init__() self.dense1 = nn.Dense(5,activation='relu') self.dense2 = nn.Dense(25,activation='relu') self.dense3 = nn.Dense(2)  
 def forward(self, x): layer1 = self.dense1(x) layer2 = self.dense2(layer1) layer3 = self.dense3(layer2) return layer3  net = MLP()  
net  
```  
  
```python  
net.dense1.params  
```  
Each layer includes parameters that are stored in a `Parameter` class. You can  
access them using the `params()` method.  
  
## Creating custom layers using Parameters (Blocks API)  
  
MXNet includes a `Parameter` method to hold your parameters in each layer. You  
can create custom layers using the `Parameter` class to include computation that  
may otherwise be not included in the built-in layers. For example, for a dense  
layer, the weights and biases will be created using the `Parameter` method. But  
if you want to add additional computation to the dense layer, you can create it  
using parameter method.  
  
Instantiate a parameter, e.g weights with a size `(5,0)` using the `shape`  
argument.  
  
```python  
from mxnet.gluon import Parameter  
  
weight = Parameter("custom_parameter_weight",shape=(5,-1))  
bias = Parameter("custom_parameter_bias",shape=(5,-1))  
  
weight,bias  
```  
  
The `Parameter` method includes a `grad_req` argument that specifies how you  
want to capture gradients for this Parameter. Under the hood, that lets gluon  
know that it has to call `.attach_grad()` on the underlying array. By default,  
the gradient is updated everytime the gradient is written to the grad  
`grad_req='write'`.  
  
Now that you know how parameters work, you are ready to create your very own  
fully-connected custom layer.  
  
To create the custom layers using parameters, you can use the same skeleton with  
`nn.Block` base class. You will create a custom dense layer that takes parameter  
x and returns computed `w*x + b` without any activation function  
  
```python  
class custom_layer(nn.Block):  
 def __init__(self,out_units,in_units=0): super().__init__() self.weight = Parameter("weight",shape=(in_units,out_units),allow_deferred_init=True) self.bias = Parameter("bias",shape=(out_units,),allow_deferred_init=True)  
 def forward(self, x): return np.dot(x, self.weight.data()) + self.bias.data()```  
  
Parameter can be instantiated before the corresponding data is instantiated. For  
example, when you instantiate a Block but the shapes of each parameter still  
need to be inferred, the Parameter will wait for the shape to be inferred before  
allocating memory.  
  
```python  
dense = custom_layer(3,in_units=5)  
dense.initialize()  
dense(np.random.uniform(size=(4, 5)))  
```  
  
Similarly, you can use the following code to implement a famous network called  
[LeNet](http://yann.lecun.com/exdb/lenet/) through `nn.Block` using the built-in  
`Dense` layer and using `custom_layer` as the last layer  
  
```python  
class LeNet(nn.Block):  
 def __init__(self): super().__init__() self.conv1  = nn.Conv2D(channels=6, kernel_size=3, activation='relu') self.pool1  = nn.MaxPool2D(pool_size=2, strides=2) self.conv2  = nn.Conv2D(channels=16, kernel_size=3, activation='relu') self.pool2  = nn.MaxPool2D(pool_size=2, strides=2) self.dense1 = nn.Dense(120, activation="relu") self.dense2 = nn.Dense(84, activation="relu") self.dense3 = nn.Dense(10)  
 def forward(self, x): x = self.conv1(x) x = self.pool1(x) x = self.conv2(x) x = self.pool2(x) x = self.dense1(x) x = self.dense2(x) x = self.dense3(x) return x  Lenet = LeNet()  
```  
  
```python  
class LeNet_custom(nn.Block):  
 def __init__(self): super().__init__() self.conv1  = nn.Conv2D(channels=6, kernel_size=3, activation='relu') self.pool1  = nn.MaxPool2D(pool_size=2, strides=2) self.conv2  = nn.Conv2D(channels=16, kernel_size=3, activation='relu') self.pool2  = nn.MaxPool2D(pool_size=2, strides=2) self.dense1 = nn.Dense(120, activation="relu") self.dense2 = nn.Dense(84, activation="relu") self.dense3 = custom_layer(10,84)  
 def forward(self, x): x = self.conv1(x) x = self.pool1(x) x = self.conv2(x) x = self.pool2(x) x = self.dense1(x) x = self.dense2(x) x = self.dense3(x) return x  Lenet_custom = LeNet_custom()  
```  
  
```python  
image_data = np.random.uniform(-1,1, (1,1,28,28))  
  
Lenet.initialize()  
Lenet_custom.initialize()  
  
print("Lenet:")  
print(Lenet(image_data))  
  
print("Custom Lenet:")  
print(Lenet_custom(image_data))  
```  
  
  
You can use `.data` method to access the weights and bias of a particular layer.  
For example, the following  accesses the first layer's weight and sixth layer's bias.  
  
```python  
Lenet.conv1.weight.data().shape, Lenet.dense1.bias.data().shape    
```  
  
## Using predefined (pretrained) architectures  
  
Till now, you have seen how to create your own neural network architectures. But  
what if you want to replicate or baseline your dataset using some of the common  
models in computer visions or natural language processing (NLP). Gluon includes  
common architectures that you can directly use. The Gluon Model Zoo provides a  
collection of off-the-shelf models e.g. RESNET, BERT etc. These architectures  
are found at:  
  
- [Gluon CV model zoo](https://gluon-cv.mxnet.io/model_zoo/index.html)  
  
- [Gluon NLP model zoo](https://gluon-nlp.mxnet.io/model_zoo/index.html)  
  
```python  
from mxnet.gluon import model_zoo  
  
net = model_zoo.vision.resnet50_v2(pretrained=True)  
net.hybridize()  
  
dummy_input = np.ones(shape=(1,3,224,224))  
output = net(dummy_input)  
output.shape  
```  
  
## Deciding the paradigm for your network  
  
In MXNet, Gluon API (Imperative programming paradigm) provides a user friendly  
way for quick prototyping, easy debugging and natural control flow for people  
familiar with python programming.  
  
However, at the backend, MXNET can also convert the network using Symbolic or  
Declarative programming into static graphs with low level optimizations on  
operators. However, static graphs are less flexible because any logic must be  
encoded into the graph as special operators like scan, while_loop and cond. It’s  
also hard to debug.  
  
So how can you make use of symbolic programming while getting the flexibility of  
imperative programming to quickly prototype and debug?  
  
Enter **HybridBlock**  
  
HybridBlocks can run in a fully imperatively way where you define their  
computation with real functions acting on real inputs. But they’re also capable  
of running symbolically, acting on placeholders. Gluon hides most of this under  
the hood so you will only need to know how it works when you want to write your  
own layers.  
  
```python  
net_hybrid_seq = nn.HybridSequential()  
  
net_hybrid_seq.add(nn.Dense(5,in_units=3,activation='relu'),  
 nn.Dense(25, activation='relu'), nn.Dense(2) )  
net_hybrid_seq  
```  
  
To compile and optimize `HybridSequential`, you can call its `hybridize` method.  
  
```python  
net_hybrid_seq.hybridize()  
```  

  
## Creating custom layers using Parameters (HybridBlocks API)  
  
When you instantiated your custom layer, you specified the input dimension  
`in_units` that initializes the weights with the shape specified by `in_units`  
and `out_units`. If you leave the shape of `in_unit` as unknown, you defer the  
shape to the first forward pass. For the custom layer, you define the  
`infer_shape()` method and let the shape be inferred at runtime.  
  
```python  
class custom_layer(nn.HybridBlock):  
 def __init__(self,out_units,in_units=-1): super().__init__() self.weight = Parameter("weight",shape=(in_units,out_units),allow_deferred_init=True) self.bias = Parameter("bias",shape=(out_units,),allow_deferred_init=True)     def forward(self, x):  
 print(self.weight.shape,self.bias.shape) return np.dot(x, self.weight.data()) + self.bias.data()     def infer_shape(self, x):  
 print(self.weight.shape,x.shape) self.weight.shape = (x.shape[-1],self.weight.shape[1])  dense = custom_layer(3)  
dense.initialize()  
dense(np.random.uniform(size=(4, 5)))  
```  
  
### Performance  
  
To get a sense of the speedup from hybridizing, you can compare the performance  
before and after hybridizing by measuring the time it takes to make 1000 forward  
passes through the network.  
  
```python  
from time import time  
  
def benchmark(net, x):  
 y = net(x) start = time() for i in range(1,1000): y = net(x) return time() - start  
x_bench = np.random.normal(size=(1,512))  
  
net_hybrid_seq = nn.HybridSequential()  
  
net_hybrid_seq.add(nn.Dense(256,activation='relu'),  
 nn.Dense(128, activation='relu'), nn.Dense(2) )net_hybrid_seq.initialize()  
  
print('Before hybridizing: %.4f sec'%(benchmark(net_hybrid_seq, x_bench)))  
net_hybrid_seq.hybridize()  
print('After hybridizing: %.4f sec'%(benchmark(net_hybrid_seq, x_bench)))  
```  
  
Peeling back another layer, you also have a `HybridBlock` which is the hybrid  
version of the `Block` API.  
  
Similar to the `Blocks` API, you define a `forward` function for `HybridBlock`  
that takes an input `x`. MXNet takes care of hybridizing the model at the  
backend so you don't have to make changes to your code to convert it to a  
symbolic paradigm.  
  
```python  
from mxnet.gluon import HybridBlock  
  
class MLP_Hybrid(HybridBlock):  
 def __init__(self): super().__init__() self.dense1 = nn.Dense(256,activation='relu') self.dense2 = nn.Dense(128,activation='relu') self.dense3 = nn.Dense(2)  
 def forward(self, x): layer1 = self.dense1(x) layer2 = self.dense2(layer1) layer3 = self.dense3(layer2) return layer3  net_Hybrid = MLP_Hybrid()  
net_Hybrid.initialize()  
  
print('Before hybridizing: %.4f sec'%(benchmark(net_Hybrid, x_bench)))  
net_Hybrid.hybridize()  
print('After hybridizing: %.4f sec'%(benchmark(net_Hybrid, x_bench)))  
```  
  
Given a HybridBlock whose forward computation consists of going through other  
HybridBlocks, you can compile that section of the network by calling the  
HybridBlocks `.hybridize()` method.  
  
All of MXNet’s predefined layers are HybridBlocks. This means that any network  
consisting entirely of predefined MXNet layers can be compiled and run at much  
faster speeds by calling `.hybridize()`.  
  
## Saving and Loading your models  
  
The Blocks API also includes saving your models during and after training so  
that you can host the model for inference or avoid training the model again from  
scratch. Another reason would be to train your model using one language (like  
Python that has a lot of tools for training) and run inference using a different  
language.  
  
There are two ways to save your model in MXNet.  
1. Save/load the model weights/parameters only  
2. Save/load the model weights/parameters and the architectures  
  
### 1. Save/load the model weights/parameters only
  
You can use `save_parameters` and `load_parameters` method to save and load the  
model weights. Take your simplest model `layer` and save your parameters first.  
The model parameters are the params that you save **after** you train your  
model.  
  
```python  
file_name = 'layer.params'  
layer.save_parameters(file_name)  
```  
  
And now load this model again. To load the parameters into a model, you will  
first have to build the model. To do this, you will need to create a simple  
function to build it.  
  
```python  
def build_model():  
 layer = nn.Dense(5, in_units=3,activation='relu') return layer  
layer_new = build_model()  
```  
  
```python  
layer_new.load_parameters('layer.params')  
```  
  
**Note**: The `save_parameters` and `load_parameters` method is used for models  
that use a `Block` method instead of  `HybridBlock` method to build the model.  
These models may have complex architectures where the model architectures may  
change during execution. E.g. if you have a model that uses an if-else  
conditional statement to choose between two different architectures.  
  
### 2. Save/load the model weights/parameters and the architectures
  
For models that use the **HybridBlock**, the model architecture stays static and  
do no change during execution. Therefore both model parameters **AND**  
architecture can be saved and loaded using `export`, `imports` methods.  
  
Now look at your `MLP_Hybrid` model and export the model using the `export`  
function. The export function will export the model architecture into a `.json`  
file and model parameters into a `.params` file.  
  
```python  
net_Hybrid.export('MLP_hybrid')  
```  
  
```python  
net_Hybrid.export('MLP_hybrid')  
```  
  
Similarly, to load this model back, you can use `gluon.nn.SymbolBlock`. To  
demonstrate that, load the network serialized above.  
  
```python  
import warnings  
with warnings.catch_warnings():  
 warnings.simplefilter("ignore") net_loaded = nn.SymbolBlock.imports("MLP_hybrid-symbol.json", ['data'], "MLP_hybrid-0000.params",ctx=None)```  
  
```python  
net_loaded(x_bench)  
```  
  
## Visualizing your models  
  
In MXNet, the `Block.Summary()` method allows you to view the block’s shape  
arguments and view the block’s parameters. When you combine multiple blocks into  
a model, the `summary()` applied on the model allows you to view each block’s  
summary, the total parameters, and the order of the blocks within the model. To  
do this the `Block.summary()` method requires one forward pass of the data,  
through your network, in order to create the graph necessary for capturing the  
corresponding shapes and parameters. Additionally, this method should be called  
before the hybridize method, since the hybridize method converts the graph into  
a symbolic one, potentially changing the operations for optimal computation.  
  
Look at the following examples  
  
- layer: our single layer network  
- Lenet: a non-hybridized LeNet network  
- net_Hybrid: our MLP Hybrid network  
  
```python  
layer.summary(x)  
```  
  
```python  
Lenet.summary(image_data)  
```  
  
You are able to print the summaries of the two networks `layer` and `Lenet`  
easily since you didn't hybridize the two networks. However, the last network  
`net_Hybrid` was hybridized above and throws an `AssertionError` if you try  
`net_Hybrid.summary(x_bench)`. To print the summary for `net_Hybrid`, call  
another instance of the same network and instantiate it for our summary and then  
hybridize it  
  
```python  
net_Hybrid_summary = MLP_Hybrid()  
  
net_Hybrid_summary.initialize()  
  
net_Hybrid_summary.summary(x_bench)  
  
net_Hybrid_summary.hybridize()  
```  
  
## Next steps:  
  
Now that you have created a neural network, learn how to automatically compute  
the gradients in [Step 3: Automatic differentiation with  
autograd](3-autograd.md).
