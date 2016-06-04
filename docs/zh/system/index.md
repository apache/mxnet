# System Design Note

这份设计文档包含了 MXNet 的相关系统设计和通用的深度学习 library. 我们相信开源设计文档, 有利用我们的听众可以更好的理解我们的动机, 系统的价值以及我们在做选择的时候的权衡. 这个可以帮助深度学习的参与者以及类似系统的设计者.


## Deep Learning Design Notes

这部分包含不同的角度来描述深度学习系统的设计文档, 包含对系统的抽象, 优化策略以及不同选择中如何做权衡.

* [深度学习编程模型](program_model.md)
* [深度学习依赖引擎](note_engine.md)
* [Squeeze the Memory Consumption of Deep Learning](note_memory.md)
* [压缩深度学习的内存开销](note_data_loading.md)
* [RNN 接口](rnn_interface.md)

下面的部分是与 MXNet 相关的.

## Overview of the Design

![System Overview](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/system/overview.png)

上面显示的是 mxnet 的主要的模块以及它们之间如何进行交互. 这些模块是

- [运行时依赖引擎](dep_engine.md): 根据操作的读写依赖关系来调度和执行这些操作.
- Storage Allocator: 可以高效的申请内存和重复利用内存, 包括 CPU 的主存和 GPU 的显存.
- Resource Manager: 管理全局资源, 包括 随机数产生器以及临时空间.
- NDArray: 动态的,异步的n维数组, 为MXNet 提供命令式编程模型.
- Symbolic Execution: 静态的符号图执行器, 提供高效地符号图的执行和优化.
- [Operator](operator.md): Operator 定义静态的 (forward) 前向计算和梯度计算 (backprop).
- [SimpleOp](operator_util.md): 统一 NDArray 的 operator 和Symbolic Operator 的扩展方式.
- Symbol Construction: 提供了构建是计算图 (网络结构配置) 的符号化构建过程.
- [KVStore](multi_node.md): Key-value 存储读写接口, 用来简化模型参数的多副本之间的同步.
- Data Loading(IO): 高效的分布式的数据读写和数据扩增.


## How to Read the Code
- 所有模块的接口在 [include](../../include) 可以找到, 这些接口有很详细的注释文档.
- 可以通过  [Doxygen Version](https://mxnet.readthedocs.org/en/latest/doxygen) 来阅读文档.
- 所有模块都只依赖其他模块在[include](../../include) 目录下的的头文件.
- 模块的具体实现在 [src](../../src) 目录下.
- 所有源码的作用域仅限于源码所在目录  [src/common](../../src/common) and [include](../../include).

大部分模块是自洽的, 仅仅依赖于 engine 模块. 所以你可以自由的挑出你感兴趣的模块来阅读.

### Analogy to CXXNet
- Symbolic Execution 可以被认为是有更多优化的神经网络的执行过程 (forward,
  backprop).
-  Operator 可以被认为是Layers, 但是需要传入 weights 和 bias 参数.
	- 它可以包含更多的接口 (可选) 来进一步的优化内存的使用量.
- 符号化的构件构成是很先进的网络模型配置文件.
- 运行时依赖引擎有些类似线程池.
	- 但是可以让我们更加容易的完成依赖追踪.
- KVStore 采用了简单的 parameter-server 接口来优化 GPU 之间的数据同步.


### Analogy to Minerva
- MXNet 中的运行时依赖引擎就是 Minerva中的 DAGEngine, 除了对可变数据有更好的支持.
- NDArray 和 owl.NDArray 是一样的, 和运行时依赖引擎一样, 加强了对可变数据的支持, 以及可以和符号操作相互交互

Documents of Each Module
------------------------
* [Runtime Dependency Engine](engine.md)
* [Operators](operator.md)
* [SimpleOp](operator_util.md)
-

List of Other Resources
-----------------------
* [Doxygen Version of C++ API](https://mxnet.readthedocs.org/en/latest/doxygen) gives a comprehensive document of C++ API.
* [Contributor Guide](../how_to/contribute.md) gives guidelines on how to push changes to the project.
