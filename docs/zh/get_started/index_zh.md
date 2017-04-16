# MXNet: 一个可扩展的深度学习框架
MXNet 是一个开源的深度学习框架，允许你在不同类型的设备上定义、训练、部署深度神经网络，从公共云服务器到移动设备均可。它具备很高的可扩展性，允许快速模型训练，支持灵活的程序模型和不同的编程语言。MXNet 为了最大化效率和生产力，允许你混合使用符号式(symbolic)和命令式(imperative)编程。MXNet 建立在一个动态依赖调度器上(dynamic dependency scheduler)，它可以同时运行符号式(symbolic)和命令式(imperative)语言。在它之上是一个图优化层(graph optimization layer)，可以让符号式语言执行的快而且内存效率高。MXNet 库是一个便携式和轻量级的库，可以适用于 GPU 集群和不同的设备。

# 设置和安装
你可以在 Amazon Linux, Ubuntu/Debian, OS X, and Windows 操作系统上运行 MXNet,同时 Docker 和云服务(比如AWS)也可以运行。MXNet 目前支持 Python, R, Julia 和 Scala 编程语言

如果你是在 Amazon Linux 或者 Ubuntu 上使用 Python/R 语言，那么你可以通过 git、bash 脚本快速安装 MXNet 库和所有相关依赖。

参考如下条目获取配置 MXNet 的更详细信息：
* [MXNet设备要求](./setup_zh.md#设备要求)
* [详细安装教程](./setup_zh.md#概述)
* [常见安装问题](./setup_zh.md#常见安装问题)

# 开始使用

当我们安装完 MXNet 和语言库以后，可以通过下面的代码来检查安装是否成功

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

Python 接口可以 `numpy.NDArray` 很相近：

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

你可以在纯 Scala 种执行张量(tensor)和矩阵(matrix)运算:

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
# 推荐教程

* [使用卷积神经网络(Convolutional Neural Networks)识别手写数字](http://mxnet.io/tutorials/python/mnist.html) (初级)
* [使用LSTMs训练字符级(Character-level)语言模型](http://mxnet.io/tutorials/python/char_lstm.html) (高级)


# 下一步
* [配置和安装](./setup_zh.md)
* [教程](http://mxnet.io/tutorials/index.html)
* [如何使用](http://mxnet.io/how_to/index.html)
* [架构设计](http://mxnet.io/architecture/index.html)


# MXNet 开源社区

**广泛的模型支持** – 训练和部署最新的 CNN 和 LSTM 模型

&nbsp;


**丰富的库和参考例子** – 包含样例教程(带源码)，比如图像分类，语言建模，神经艺术(neural art),语音识别等等。

&nbsp;


**开放协同的社区** – 顶级大学和商业伙伴的支持和贡献


&nbsp;
