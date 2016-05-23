The following is an overview of MXNet in Chinese. For english readers, please
refer to our [NIPS learnsys paper](http://learningsys.org/papers/LearningSys_2015_paper_1.pdf)

# MXNet设计和实现简介

神经网络本质上是一种语言，我们通过它来表达对应用问题的理解。例如我们用卷积层来表达空间相关性，RNN来表达时间连续性。根据问题的复杂性和信息如何从输入到输出一步步提取，我们将不同大小的层按一定原则连接起来。近年来随着数据的激增和计算能力的大幅提升，神经网络也变得越来越深和大。例如最近几次imagnet竞赛的冠军都使用有数十至百层的网络。对于这一类神经网络我们通常称之为深度学习。从应用的角度而言，对深度学习最重要的是如何方便地表述神经网络，以及如何快速训练得到模型。

对于一个优秀的深度学习系统，或者更广来说优秀的科学计算系统，最重要的是编程接口的设计。他们都采用将一个*领域特定语言(domain specific language)*嵌入到一个主语言中。例如numpy将矩阵运算嵌入到python中。这类嵌入一般分为两种，其中一种嵌入的较浅，其中每个语句都按原来的意思执行，且通常采用*命令式编程(imperative programming)*，其中numpy和Torch就是属于这种。而另一种则用一种深的嵌入方式，提供一整套针对具体应用的迷你语言。这一种通常使用*声明式语言(declarative programing)*，既用户只需要声明要做什么，而具体执行则由系统完成。这类系统包括Caffe，theano和刚公布的TensorFlow。

这两种方式各有利弊，总结如下

- 命令式编程:
    - **如何执行 *a=b+1***: 需要b已经被赋值。立即执行加法，将结果保存在a中。
    - **优点**: 语义上容易理解，灵活，可以精确控制行为。通常可以无缝地和主语言交互，方便地利用主语言的各类算法，工具包，debug和性能调试器。
    - **缺点**: 实现统一的辅助函数和提供整体优化都很困难。
- 声明式编程:
    - **如何执行 *a=b+1***: 返回对应的计算图(computation graph)，我们可以之后对b进行赋值，然后再执行加法运算
    - **优点**: 在真正开始计算的时候已经拿到了整个计算图，所以我们可以做一系列优化来提升性能。实现辅助函数也容易，例如对任何计算图都提供forward和backward函数，对计算图进行可视化，将图保存到硬盘和从硬盘读取。
    - **缺点**: 很多主语言的特性都用不上。某些在主语言中实现简单，但在这里却经常麻烦，例如if-else语句 。debug也不容易，例如监视一个复杂的计算图中的某个节点的中间结果并不简单。

目前现有的系统大部分都采用上两种编程模式的一种。与它们不同的是，MXNet尝试将两种模式无缝的结合起来。在命令式编程上MXNet提供张量运算，而声明式编程中MXNet支持符号表达式。用户可以自由的混合它们来快速实现自己的想法。例如我们可以用声明式编程来描述神经网络，并利用系统提供的自动求导来训练模型。另一方便，模型的迭代训练和更新模型法则中可能涉及大量的控制逻辑，因此我们可以用命令式编程来实现。同时我们用它来进行方便地调式和与主语言交互数据。


下表我们比较MXNet和其他流行的深度学习系统

```eval_rst
======= =======  ========== ======== ================== ==========
框架    Caffe     Torch      Theano    TensorFlow         MXNet
主语言  C++       Lua        Python     C++                C++
从语言  Python,     x         x         Python             Python, R, Julia, Scala, Javascript, Matlab, Go
        Matlab
硬件    CPU,      CPU, GPU, CPU,      CPU, GPU, mobile     CPU, GPU,mobile
        GPU,      FPGA      GPU,
分布式   x        x           x           v                  v
命令式   x        v           v           x                  v
声明式   v        x             x          v                   v
======= =======  ========== ======== ================== ==========
```


MXNet的系统架构如下图所示：

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/paper/sys.png width=300/>

从上到下分别为各种主语言的嵌入，编程接口（矩阵运算，符号表达式，分布式通讯），两种编程模式的统一系统实现，以及各硬件的支持。接下一章我们将介绍编程接口，然后下一章介绍系统实现。之后我们给出一些实验对比结果，以及讨论MXNet的未来。

## 编程接口

### `Symbol`： 声明式的符号表达式

MXNet使用多值输出的符号表达式来声明计算图。符号是由操作子构建而来。一个操作子可以是一个简单的矩阵运算“+”，也可以是一个复杂的神经网络里面的层，例如卷积层。一个操作子可以有多个输入变量和多个输出变量，还可以有内部状态变量。一个变量既可以是自由的，我们可以之后对其赋值；也可以是某个操作子的输出。例如下面的代码中我们使用Julia来定义一个多层感知机，它由一个代表输入数据的自由变量，和多个神经网络层串联而成。

```julia
using MXNet
mlp = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=64) =>
  mx.Activation(act_type=:relu)    =>
  mx.FullyConnected(num_hidden=10) =>
  mx.Softmax()
```

在执行一个符号表达式前，我们需要对所有的自由变量进行赋值。上例中，我们需要给定数据，和各个层里隐式定义的输入，例如全连接层的权重和偏值。我们同时要申明所需要的输出，例如softmax的输出。

除了执行获得softmax输出外（通常也叫forward），符号表达式也支持自动求导来获取各权重和偏值对应的梯度（也称之为backward）。此外，我们还可以提前估计计算时需要的内存，符号表达式的可视化，读入和输出等。

### `NDArray`：命令式的张量计算

MXNet提供命令式的张量计算来桥接主语言的和符号表达式。下面代码中，我们在GPU上计算矩阵和常量的乘法，并使用numpy来打印结果

```python
>>> import MXNet as mx
>>> a = mx.nd.ones((2, 3),
... mx.gpu())
>>> print (a * 2).asnumpy()
[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

另一方面，NDArray可以无缝和符号表达式进行对接。假设我们使用Symbol定义了一个神经网络，那么我们可以如下实现一个梯度下降算法

```c++
for (int i = 0; i < n; ++i) {
  net.forward();
  net.backward();
  net.weight -= eta * net.grad
}
```

这里梯度由Symbol计算而得。Symbol的输出结果均表示成NDArray，我们可以通过NDArray提供的张量计算来更新权重。此外，我们还利用了主语言的for循环来进行迭代，学习率eta也是在主语言中进行修改。

上面的混合实现跟使用纯符号表达式实现的性能相差无二，然后后者在表达控制逻辑时会更加复杂。其原因是NDArray的执行会和Symbol类似的构建一个计算图，并与其他运算一同交由后台引擎执行。对于运算`-=`由于我们只是将其结果交给另一个Symbol的forward作为输入，因此我们不需要立即得到结果。当上面的for循环结束时，我们只是将数个Symbol和NDarray对应的计算图提交给了后台引擎。当我们最终需要结果的时候，例如将weight复制到主语言中或者保存到磁盘时，程序才会被阻塞直到所有计算完成。

### `KVStore`：多设备间的数据交互

MXNet提供一个分布式的key-value存储来进行数据交换。它主要有两个函数，
1. push： 将key-value对从一个设备push进存储
2. pull：将某个key上的值从存储中pull出来此外，KVStore还接受自定义的更新函数来控制收到的值如何写入到存储中。最后KVStore提供数种包含最终一致性模型和顺序一致性模型在内的数据一致性模型。

在下面例子中，我们将前面的梯度下降算法改成分布式梯度下降。

```c++
KVStore kv("dist_async");
kv.set_updater([](NDArray w, NDArray g) {
    w -= eta * g;
  });
for (int i = 0; i < max_iter; ++i) {
   kv.pull(net.weight);
   net.forward();
   net.backward();
   kv.push(net.grad);
}
```

在这里我们先使用最终一致性模型创建一个kvstore，然后将更新函数注册进去。在每轮迭代前，每个计算节点先将最新的权重pull回来，之后将计算的得到的梯度push出去。kvstore将会利用更新函数来使用收到的梯度更新其所存储的权重。

这里push和pull跟NDArray一样使用了延后计算的技术。它们只是将对应的操作提交给后台引擎，而引擎则调度实际的数据交互。所以上述的实现跟我们使用纯符号实现的性能相差无几。

### 读入数据模块

数据读取在整体系统性能上占重要地位。MXNet提供工具能将任意大小的样本压缩打包成单个或者数个文件来加速顺序和随机读取。

通常数据存在本地磁盘或者远端的分布式文件系统上（例如HDFS或者Amazon S3)，每次我们只需要将当前需要的数据读进内存。MXNet提供迭代器可以按块读取不同格式的文件。迭代器使用多线程来解码数据，并使用多线程预读取来隐藏文件读取的开销。

### 训练模块

MXNet实现了常用的优化算法来训练模型。用户只需要提供数据数据迭代器和神经网络的Symbol便可。此外，用户可以提供额外的KVStore来进行分布式的训练。例如下面代码使用分布式异步SGD来训练一个模型，其中每个计算节点使用两块GPU。

```python
import MXNet as mx
model = mx.model.FeedForward(
    ctx                = [mx.gpu(0), mx.gpu(1)],
    symbol             = network,
    num_epoch          = 100,
    learning_rate      = 0.01,
    momentum           = 0.9,
    wd                 = 0.00001,
    initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34))
model.fit(
    X                  = train_iter,
    eval_data          = val_iter,
    kvstore            = mx.kvstore.create('dist_async'),
    epoch_end_callback = mx.callback.do_checkpoint('model_'))
```

## 系统实现

### 计算图

一个已经赋值的符号表达式可以表示成一个计算图。下图是之前定义的多层感知机的部分计算图，包含forward和backward。

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/paper/graph.png width=300/>

其中圆表示变量，方框表示操作子，箭头表示数据依赖关系。在执行之前，MXNet会对计算图进行优化，以及为所有变量提前申请空间。

#### 计算图优化

计算图优化已经在数据库等领域被研究多年，我们目前只探索了数个简单的方法。
1. 注意到我们提前申明了哪些输出变量是需要的，这样我们只需要计算这些输出需要的操作。例如，在预测时我们不需要计算梯度，所以整个backforward图都可以忽略。而在特征抽取中，我们可能只需要某些中间层的输出，从而可以忽略掉后面的计算。
2. 我们可以合并某些操作。例如 *a*b+1*只需要一个blas或者cuda函数即可，而不需要将其表示成两个操作。
3. 我们实现了一些“大”操作，例如一个卷积层就只需要一个操作子。这样我们可以大大减小计算图的大小，并且方便手动的对这个操作进行优化。

#### 内存申请

内存通常是一个重要的瓶颈，尤其是对GPU和智能设备而言。而神经网络计算时通常需要大量的临时空间，例如每个层的输入和输出变量。对每个变量都申请一段独立的空间会带来高额的内存开销。幸运的是，我们可以从计算图推断出所有变量的生存期，就是这个变量从创建到最后被使用的时间段，从而可以对两个不交叉的变量重复使用同一内存空间。这个问题在诸多领域，例如编译器的寄存器分配上，有过研究。然而最优的分配算法需要 *O(n<sup>2</sup>)* 时间复杂度，这里n是图中变量的个数。

MXNet提供了两个启发式的策略，每个策略都是线性的复杂度。
1. inplace。在这个策略里，我们模拟图的遍历过程，并为每个变量维护一个还有多少其他变量需要它的计数。当我们发现某个变量的计数变成0时，我们便回收其内存空间。
2. co-share。我们允许两个变量使用同一段内存空间。这么做当然会使得这两个变量不能同时在写这段空间。所以我们只考虑对不能并行的变量进行co-share。每一次我们考虑图中的一条路（path），路上所有变量都有依赖关系所以不能被并行，然后我们对其进行内存分配并将它们从图中删掉。

### 引擎

在MXNet中，所有的任务，包括张量计算，symbol执行，数据通讯，都会交由引擎来执行。首先，所有的资源单元，例如NDArray，随机数生成器，和临时空间，都会在引擎处注册一个唯一的标签。然后每个提交给引擎的任务都会标明它所需要的资源标签。引擎则会跟踪每个资源，如果某个任务所需要的资源到到位了，例如产生这个资源的上一个任务已经完成了，那么引擎会则调度和执行这个任务。

通常一个MXNet运行实例会使用多个硬件资源，包括CPU，GPU，PCIe通道，网络，和磁盘，所以引擎会使用多线程来调度，既任何两个没有资源依赖冲突的任务都可能会被并行执行，以求最大化资源利用。

与通常的数据流引擎不同的是，MXNet的引擎允许一个任务修改现有的资源。为了保证调度正确性，提交任务时需要分开标明哪些资源是只读，哪些资源会被修改。这个附加的写依赖可以带来很多便利。例如我们可以方便实现在numpy以及其他张量库中常见的数组修改操作，同时也使得内存分配时更加容易，比如操作子可以修改其内部状态变量而不需要每次都重来内存。再次，假如我们要用同一个种子生成两个随机数，那么我们可以标注这两个操作会同时修改种子来使得引擎不会并行执行，从而使得代码的结果可以很好的被重复。

### 数据通讯

KVStore的实现是基于参数服务器。但它跟前面的工作有两个显著的区别。
1. 我们通过引擎来管理数据一致性，这使得参数服务器的实现变得相当简单，同时使得KVStore的运算可以无缝的与其他结合在一起。
2. 我们使用一个两层的通讯结构，原理如下图所示。第一层的服务器管理单机内部的多个设备之间的通讯。第二层服务器则管理机器之间通过网络的通讯。第一层的服务器在与第二层通讯前可能合并设备之间的数据来降低网络带宽消费。同时考虑到机器内和外通讯带宽和延时的不同性，我们可以对其使用不同的一致性模型。例如第一层我们用强的一致性模型，而第二层我们则使用弱的一致性模型来减少同步开销。

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/paper/ps.png width=300/>

### 可移植性

轻量和可移植性是MXNet的一个重要目标。MXNet核心使用C++实现，并提供C风格的头文件。因此方便系统移植，也使得其很容易被其他支持C FFI (forigen language interface )的语言调用。此外，我们也提供一个脚本将MXNet核心功能的代码连同所有依赖打包成一个单一的只有数万行的C++源文件，使得其在一些受限的平台，例如智能设备，方便编译和使用。

## 实验结果

这里我们提供一些早期的实验结果。

### 与其他系统相比

我们首先使用一个流行卷积网络测试方案来对比MXNet与Torch，Caffe和TensorFlow在过去几届imagenet竞赛冠军网络上的性能。每个系统使用同样的CUDA 7.0和CUDNN 3，但TensorFlow使用其只支持的CUDA 6.5 和CUDNN 2。我们使用单块GTX 980并报告单个forward和backward的耗时。

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/paper/time_forward_backward.png width=400/>

可以看出MXNet，Torch和Caffe三者在性能上不相上下。这个符合预期，因为在单卡上我们评测的几个网络的绝大部分运算都由CUDA和CUDNN完成。TensorFlow比其他三者都慢2倍以上，这可能由于是低版本的CUDNN和项目刚开源的缘故。

### 内存的使用

接下来我们考察不同的内存分配算法对内存占用的影响。下图分别表示使用batch=128时，在做预测时和做训练时的不同算法在内部变量（除去模型，最初输入和最终输出）上的内存开销。

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/paper/mem_forward.png width=400/> <img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/paper/mem_forward_backward.png width=400/>

可以看出，inplace和co-share两者都可以极大的降低内存使用。将两者合起来可以在训练时减少2倍内存使用，在预测时则可以减小4倍内存使用。特别的，即使是最复杂的vggnet，对单张图片进行预测时，MXNet只需要16MB额外内存。

### Scalability

最后我们报告在分布式训练下的性能。我们使用imagenet 1k数据（120万224x224x3图片，1000类），并用googlenet加上batch normalization来训练。我们使用Amazon EC2 g2.8x，单机和多机均使用同样的参数，下图表示了使用单机和10台g2.8x时的收敛情况。

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/inception-with-bn-imagnet1k.png width=600px/>

从训练精度来看，单机的收敛比多机快，这个符合预期，因为多机时有效的batch大小比单机要大，在处理同样多的数据上收敛通常会慢。但有意思的是两者在测试精度上非常相似。

单机下每遍历一次数据需要1万4千秒，而在十台机器上，每次只需要1千4百秒。如果考虑运行时间对比测试精度，10台机器带来了10倍的提升。

## 过去，现状，和未来

大半年前我们拉来数个优秀的C++机器学习系统的开发人员成立了DMLC，本意是更方便共享各自项目的代码，并给用户提供一致的体验。当时我们有两个深度学习的项目，一个是CXXNet，其通过配置来定义和训练神经网络。另一个是Minerva，提供类似numpy一样的张量计算接口。前者在图片分类等使用卷积网络上很方便，而后者更灵活。那时候我们想能不能做一个两者功能都具备的系统，于是这样就有了MXNet。其名字来自Minerva的M和CXXNet的XNet。其中Symbol的想法来自CXXNet，而NDArray的想法来自Minerva。我们也常把MXNet叫“mix net”。

MXNet是DMLC第一个结合了所有成员努力的项目，也同时吸引了很多核心成员的加入。MXNet的目的是做一个有意思的系统，能够让大家用着方便的系统，一个轻量的和可以快速测试系统和算法想法的系统。对于未来，我们主要关注下面四个方向：

1. 支持更多的硬件，我们目前在积极考虑支持AMD GPU，高通GPU，Intel Phi，FPGA，和更多智能设备。相信MXNet的轻量和内存节省可以在这些上大有作为。
2. 更加完善的操作子。目前不论是Symbol还是NDArray支持的操作还是有限，我们希望能够尽快的扩充他们。
3. 更多编程语言。除了C++，目前MXNet对Python，R和Julia的支持比较完善。但我们希望还能有很多的语言，例如javascript。
4. 更多的应用。我们之前花了很多精力在图片分类上，下面我们会考虑很多的应用。例如上周我们试了下如何利用一张图片的风格和一张图片的内容合成一张新图片。下图是利用我办公室窗景和梵高的starry night来合成图片

 <img src=https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/input/IMG_4343.jpg width=300px><img src=https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/input/starry_night.jpg width=300px>

 <img src=https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/output/4343_starry_night.jpg width=600px>

 接下来我们希望能够在更多应用，例如语音、翻译、问答上有所产出。

我们忠心希望MXNet能为大家做深度学习相关研究和应用带来便利。也希望能与更多的开发者一起学习和进步。

## 扩展阅读

1. 此文大部分内容已经发表在NIPS LearningSys 2015上，[paper link](http://www.cs.cmu.edu/~muli/file/MXNet-learning-sys.pdf)
2. 本文只是对MXNet各个部件做了初步的介绍，更多文档参见 [MXNet/doc](http://MXNet.readthedocs.org/en/latest/index.html)
3. 本文实验代码均在 [MXNet/example](https://github.com/dmlc/mxnet/tree/master/example)
