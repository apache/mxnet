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

# Introduction
This crash course will give you a quick overview of  basics of MXNet, the core concept of NDArray (manipulating multiple dimensional arrays) and Gluon (create and train neural networks on CPU and GPU). The intended audience for this crash course is someone who are already familiar with machine learning or other deep learning frameworks.For a deep dive into the details of MXNet and deep learning architectures, please refer to [Dive Into Deep learning](http://d2l.ai/) textbook or [Introduction to Deep Learning Course](https://courses.d2l.ai/berkeley-stat-157/index.html)


# What is MXNet

Apache MXNet is a fully featured, flexibly programmable, and ultra-scalable open-source deep learning framework supporting state of the art deep learning models, including convolutional neural networks (CNNs) and long short-term memory networks (LSTMs). MXNet has its roots in academia and came about through the collaboration and contributions of researchers at several top universities  and the preferred choice of AWS (Amazon Web Services), as well as many colleges and companies.

Some key features of MXNet:
1.  **Fast and Scalable:** Easily supports multiple GPU's and distributed multi-host jobs. 
2.  **Multiple Programming language support:**  Python, Scala,  R, Java, C++, Julia, Matlab, JavaScript and Go interfaces.
3.  **Supported:** Backed by Apache Software Foundation and supported by Amazon Web Services (AWS), Microsoft Azure and highly active open-source community.
4. **Portable:** Supports an efficient deployment of a trained model for inference   on wide range of hardware configurations across various platforms of choice i.e.  low end devices, internet of things devices , serverless computing and containers.
5. **Flexible:** Supports both imperative and symbolic programming.

# Gluon

Gluon is an imperative high-level front end API in MXNet for deep learning that’s flexible and easy-to-use which comes with a lot of great features, and it can provide you everything you need: from experimentation to deploying the model without sacrificing training speed. Gluon provides S:tate of the Art models for many of the standard tasks such as Classification, Object Detection, Segmentation, etc. In one of the next sections of the tutorial, you will walk through a common use case on how to build a model using gluon, train it on your data, and deploy it for inference.


# Basic building blocks

## Tensors

Tensors  give us a generic way of describing 𝑛n-dimensional arrays with an arbitrary number of axes.Vectors, for example, are first-order tensors, and matrices are second-order tensors.  Tensors with more than two orders(axes) do not have special mathematical names.The [ndarray](https://mxnet.apache.org/versions/1.7/api/python/docs/api/ndarray/index.html) package  MXNet provides tensor implementation. Tensor class in MXNet is similar to NumPy's ndarray with additional features. First, MXNet’s `NDArray` supports fast execution on a wide range of hardware configurations, including CPU, GPU, and multi-GPU machines where as NumPy only supports CPU computation. Second, MXNet’s `NDArray` executes code lazily, allowing it to automatically parallelize multiple operations across the available hardware.

  To start, you can use arange to create a row vector x containing the first 24 integers starting with 0, though they are created as floats by default. Each of the values in a tensor is called an element of the tensor. For instance, there are 24 elements in the tensor x. Unless otherwise specified, a new tensor will be stored in main memory and designated for CPU-based computation by default. You can query the device where the tensor is located.

```{.python .input n=1}

from mxnet import np, npx

npx.set_np() # Activate NumPy-like mode.

```

```{.python .input n=2}
X = np.arange(24).reshape(2, 3, 4)
X
```
Output:
```{.python .input n=2}
array([[[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]],

       [[12., 13., 14., 15.],
        [16., 17., 18., 19.],
        [20., 21., 22., 23.]]])
```
```{.python .input n=3}
X.ctx
```
Output:
```{.python .input n=3}
cpu(0)
```

Please refer to the chapter  Use GPUs in tutorial to save and access tensors on GPUs using MXNet

# Computing paradigms

## Block
Neural network designs like [ResNet-152](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) have a fair degree of regularity. They consist of _blocks_ of repeated (or at least similarly designed) layers; these blocks then form the basis of more complex network designs. A block is a single layer, a component consisting of multiple layers, or the entire model of  complex neural network itself! One benefit of working with the block abstraction is that they can be combined into larger artifacts, often recursively. By defining code to generate blocks of arbitrary complexity on demand, you can write surprisingly compact code and still implement complex neural networks.


From a programming standpoint, a block is represented by a class and [Block](https://mxnet.apache.org/versions/1.7/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Block)  is the base class for all neural networks layers in MXNet. Any subclass of it must define a forward propagation function that transforms its input into output and must store any necessary parameters if required.
The following code generates a network with first fully-connected hidden layer with 256 units and ReLU activation, second fully connected hidden layer with 128 units and ReLU activation followed by a fully-connected output layer with 2 units (no activation function).

```{.python .input n=2}
from mxnet import nd, npx
from mxnet.gluon import nn
import time
npx.reset_np()

def get_net():
    net = nn.Sequential() 
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)
```

Output:
```{.python .input n=2}
[[-0.03025527  0.01982136]]
<NDArray 1x2 @cpu(0)>
```

## HybridBlock

Imperative and symbolic  programming represents two styles or paradigms of deep learning programming interface and historically most deep learning frameworks choose either imperative or symbolic programming. For example, both Theano and TensorFlow (inspired by the latter) make use of symbolic programming, while Chainer and its predecessor PyTorch utilize imperative programming. 

The differences bet
en imperative (interpreted) programming and symbolic programming are as follows:

— Imperative programming is easier. When imperative programming is used in Python, the majority of the code is straightforward and easy to write. It is also easier to debug imperative programming code. This is because it is easier to obtain and print all relevant intermediate variable values, or use Pythonʼs built-in debugging tools.
    
— Symbolic programming is more efficient and easier to port. It makes it easier to optimize the code during compilation, while also having the ability to port the program into a format independent of Python. This allows the program to be run in a non-Python environment, thus avoiding any potential performance issues related to the Python interpreter.

You can learn more about the difference between symbolic vs. imperative programming from this [deep learning programming paradigm](https://mxnet.apache.org/versions/1.6/api/architecture/program_model) article

When designing Gluon, developers considered whether it was possible to harness the benefits of both imperative and symbolic programming. The developers believed that users should be able to develop and debug using pure imperative programming, while having the ability to convert most programs into symbolic programming to be run when product-level computing performance and deployment are required. This was achieved by Gluon through the introduction of hybrid programming.

In hybrid programming, you can build models using either the [HybridBlock](https://mxnet.apache.org/versions/1.7/api/python/docs/api/gluon/hybrid_block.html) or the [HybridSequential](https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.HybridSequential) and [HybridConcurrent](https://mxnet.incubator.apache.org/versions/1.7/api/python/docs/api/gluon/contrib/index.html#mxnet.gluon.contrib.nn.HybridConcurrent) classes. By default, they are executed in the same way Block or Sequential  and Concurrent  classes are executed in imperative programming. When the  `hybridize`  function is called, Gluon will convert the program’s execution into the style used in symbolic programming. This allows one to optimize the compute-intensive components without sacrifices in the way a model is implemented. In fact, most models can make use of hybrid programming’s execution style.

```{.python .input n=2}
from mxnet import nd, npx
from mxnet.gluon import nn
import time
npx.reset_np()

def get_net():
    net = nn.HybridSequential()  # Here you use the class HybridSequential.
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)
```

Output:
```{.python .input n=2}
[[0.11592434 0.09331231]]
<NDArray 1x2 @cpu(0)>
```

You can call `hybridize` function to compile and optimize the `HybridSequential` in the MLP. The modelʼs computation result remains unchanged.

```{.python .input n=2}
net.hybridize()
net(x)
```
Output:
```{.python .input n=2}
[[0.11592434 0.09331231]]
<NDArray 1x2 @cpu(0)>
```



# References
1.  [Dive into Deep Learning](http://d2l.ai/) 
