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


## About MXNet

Apache MXNet is an open-source deep learning framework that provides a comprehensive and flexible API to create deep learning models. Some of the key features of MXNet are:

1.  **Fast and Scalable:** Easily supports multiple GPU's and distributed multi-host jobs. 
2.  **Multiple Programming language support:**  Python, Scala,  R, Java, C++, Julia, Matlab, JavaScript and Go interfaces. 
3.  **Supported:** Backed by Apache Software Foundation and supported by Amazon Web Services (AWS), Microsoft Azure and highly active open-source community.
4.  **Portable:** Supports an efficient deployment on a wide range of hardware configurations and platforms i.e.  low end devices, internet of things devices, serverless computing and containers.
5.  **Flexible:** Supports both imperative and symbolic programming.


### Basic building blocks

#### Tensors A.K.A Arrays

Tensors give us a generic way of describing $n$-dimensional **arrays** with an arbitrary number of axes. Vectors, for example, are first-order tensors, and matrices are second-order tensors. Tensors with more than two orders(axes) do not have special mathematical names. The [ndarray](https://mxnet.apache.org/versions/1.7/api/python/docs/api/ndarray/index.html) package in MXNet provides a tensor implementation. This class is similar to NumPy's ndarray with additional features. First, MXNet’s `NDArray` supports fast execution on a wide range of hardware configurations, including CPU, GPU, and multi-GPU machines where as NumPy only supports CPU computation. Second, MXNet’s `NDArray` executes code lazily, allowing it to automatically parallelize multiple operations across the available hardware.

You will get familiar to arrays in the [next section](1-nparray.md) of this crash course.

### Computing paradigms

#### Block

Neural network designs like [ResNet-152](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) have a fair degree of regularity. They consist of _blocks_ of repeated (or at least similarly designed) layers; these blocks then form the basis of more complex network designs. A block can be a single layer, a component consisting of multiple layers, or the entire complex neural network itself! One benefit of working with the block abstraction is that you can combine blocks into larger artifacts; often recursively. By defining code to generate blocks of arbitrary complexity on demand, you can write surprisingly compact code and still implement complex neural networks.


From a programming standpoint, a block is represented by a class and [Block](https://mxnet.apache.org/versions/1.7/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.Block)  is the base class for all neural networks layers in MXNet. Any subclass of it must define a forward propagation function that transforms its input into output and must store any necessary parameters if required.

You will see more about blocks in [Array](1-nparray.md) and [Create neural network](2-create-nn.md) sections.

#### HybridBlock

Imperative and symbolic  programming represents two styles or paradigms of deep learning programming interface and historically most deep learning frameworks choose either imperative or symbolic programming. For example, both Theano and TensorFlow (inspired by the latter) make use of symbolic programming, while Chainer and its predecessor PyTorch utilize imperative programming. 

The differences between imperative (interpreted) and symbolic programming are as follows:

* __Imperative programming__ is easier. When imperative programming is used in Python, the majority of the code is straightforward and easy to write. It is also easier to debug imperative programming code. This is because it is easier to obtain and print all relevant intermediate variable values, or use Pythonʼs built-in debugging tools.
    
* __Symbolic programming__ is more efficient and easier to port. It makes it easier to optimize the code during compilation, while also having the ability to port the program into a format independent of Python. This allows the program to be run in a non-Python environment, thus avoiding any potential performance issues related to the Python interpreter.

You can learn more about the difference between symbolic vs. imperative programming from this [deep learning programming paradigm](https://mxnet.apache.org/versions/1.6/api/architecture/program_model) article

When designing MXNet, developers considered whether it was possible to harness the benefits of both imperative and symbolic programming. The developers believed that users should be able to develop and debug using pure imperative programming, while having the ability to convert most programs into symbolic programming to be run when product-level computing performance and deployment are required. 

In hybrid programming, you can build models using either the [HybridBlock](https://mxnet.apache.org/versions/1.7/api/python/docs/api/gluon/hybrid_block.html) or the [HybridSequential](https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html#mxnet.gluon.nn.HybridSequential) and [HybridConcurrent](https://mxnet.incubator.apache.org/versions/1.7/api/python/docs/api/gluon/contrib/index.html#mxnet.gluon.contrib.nn.HybridConcurrent) classes. By default, they are executed in the same way Block or Sequential  and Concurrent  classes are executed in imperative programming. When the  `hybridize`  function is called, Gluon will convert the program’s execution into the style used in symbolic programming. This allows one to optimize the compute-intensive components without sacrifices in the way a model is implemented. In fact, most models can make use of hybrid programming’s execution style.

You will learn more about hybrid blocks and use them in the upcoming sections of the course.

### Gluon

Gluon is an imperative high-level front end API in MXNet for deep learning that’s flexible and easy-to-use which comes with a lot of great features, and it can provide you everything you need: from experimentation to deploying the model without sacrificing training speed. This is because, as discussed above, you have access to both imperative and symbolic APIs through the introduction of hybrid programming. Gluon provides State of the Art models for many of the standard tasks such as Classification, Object Detection, Segmentation, etc. In one of the next sections of the tutorial, you will walk through an example of how to build a model using gluon, train it on a dataset, and make predictions with it.

## Next steps

Dive deeper on [array representations](1-nparray.md) in MXNet.

## References
1.  [Dive into Deep Learning](http://d2l.ai/) 
