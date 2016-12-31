# MXNet: A Scalable Deep Learning Framework
MXNet is an open-source deep learning framework that allows you to define, train, and deploy deep neural networks on a wide array of devices, from cloud infrastructure to mobile devices.
It is highly scalable, allowing for fast model training, and supports a flexible programming model and multiple languages. MXNet allows you to mix symbolic and imperative programming flavors to maximize both efficiency and productivity.
MXNet is built on a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient. The MXNet library is portable and lightweight, and it scales to multiple GPUs and multiple machines.


&nbsp;

**Flexible Programming Model**
Supports both imperative and symbolic programming, maximizing efficiency and productivity


&nbsp;

**Portable from the Cloud to the Client**
Runs on CPUs or GPUs, and on clusters, servers, desktops, or mobile phones


&nbsp;

**Multiple Languages**
Supports building and training models in Python, R, Scala, Julia, and C++.  Pre-trained models can be used for prediction in even more languages like Matlab or Javascript.


&nbsp;

**Native Distributed Training**
Supports distributed training on multiple CPU/GPU machines to take advantage of cloud scale


&nbsp;

**Performance Optimized**
Parallelizes both I/O and computation with an optimized C++ backend engine, and performs optimally no matter which language you program in


&nbsp;

# MXNet Open Source Community

**Broad Model Support** – Train and deploy the latest deep convolutional neural networks (CNNs) and long short-term memory (LSTMs) models


&nbsp;

**Extensive Library of Reference Examples** – Build on sample tutorials (with code), such as image classification, language modeling, neural art, and speech recognition, and more.  


&nbsp;

**Open and Collaborative Community** – Support and contributions from many top tier universities and industry partners


&nbsp;
# Setup and Installation
You can run MXNet on Amazon Linux, Ubuntu/Debian, OS X, and Windows operating systems. MXNet can also be run on Docker and on Cloud like AWS. MXNet currently supports the Python, R, Julia and Scala languages.

If you are running Python/R on Amazon Linux or Ubuntu, you can use Git Bash scripts to quickly install the MXNet libraries and all its dependencies.

Refer below for more details on setting up MXNet:
* [Prerequisites for using MXNet](http://mxnet.io/get_started/setup.html#prerequisites)
* [Step by step instruction guide for installing MXNet](http://mxnet.io/get_started/setup.html#overview)
* [Common installation problems](http://mxnet.io/get_started/setup.html#common-installation-problems)

# Starting with the Basics | Tensor Computation

Now let's take a look at the tensor computation interface. The tensor computation interface is often more
flexible than the symbolic interface. It is often used to
implement the layers, define weight updating rules, and debug.

## Julia
```julia
julia> using MXNet

julia> a = mx.ones((2,3), mx.gpu())
mx.NDArray{Float32}(2,3)

julia> Array{Float32}(a * 2)
2×3 Array{Float32,2}:
 2.0  2.0  2.0
 2.0  2.0  2.0
```

## Python

The Python interface is similar to `numpy.NDArray`:

 ```python
    >>> import mxnet as mx
    >>> a = mx.nd.ones((2, 3), mx.gpu())
    >>> print ((a * 2).asnumpy())
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
 ```

## R

 ```r
    > require(mxnet)
    Loading required package: mxnet
    > a <- mx.nd.ones(c(2,3))
    > a
         [,1] [,2] [,3]
    [1,]    1    1    1
    [2,]    1    1    1
    > a + 1
         [,1] [,2] [,3]
    [1,]    2    2    2
    [2,]    2    2    2
 ```

## Scala

You can perform tensor or matrix computation in pure Scala:

 ```scala
    scala> import ml.dmlc.mxnet._
    import ml.dmlc.mxnet._

    scala> val arr = NDArray.ones(2, 3)
    arr: ml.dmlc.mxnet.NDArray = ml.dmlc.mxnet.NDArray@f5e74790

    scala> arr.shape
    res0: ml.dmlc.mxnet.Shape = (2,3)

    scala> (arr * 2).toArray
    res2: Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)

    scala> (arr * 2).shape
    res3: ml.dmlc.mxnet.Shape = (2,3)
 ```
# Recommended Starting Tutorials

* [Handwritten Digit Recognition using Convolutional Neural Networks](http://mxnet.io/tutorials/python/mnist.html) (Beginner)
* [Character-level language models using LSTMs](http://mxnet.io/tutorials/python/char_lstm.html) (Advanced)

# Next Steps
* [Setup and Installation](http://mxnet.io/get_started/setup.html)
* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
