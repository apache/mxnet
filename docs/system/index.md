# System Design Note

This design document contains notes that are relevant to the MXNet system design and deep learning
libraries in general. We believe that open sourcing this system design note can help general audiences understand the  motivations, benefits and drawbacks of our design choices. This will help deep learning practitioners as well as builders of other deep learning systems.

## Deep Learning Design Notes

This section will be updated with self-contained design notes on various aspect of deep learning systems,
in terms of abstraction, optimization and trade-offs.

* [Programming Models for Deep Learning](program_model.md)
* [Dependency Engine for Deep Learning](note_engine.md)
* [Squeeze the Memory Consumption of Deep Learning](note_memory.md)
* [Efficient Data Loading Module for Deep Learning](note_data_loading.md)
* [Survay of RNN Interface](rnn_interface.md)

The next parts will be specific to MXNet

## Overview of the Design

![System Overview](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/system/overview.png)

The above shows major modules of mxnet, and how do they interact with each
other. The modules are
- [Runtime Dependency Engine](engine.md): Schedules and executes the
  operations according to their read/write dependency.
- Storage Allocator: Efficiently allocate and recycles memory blocks for GPU and
  CPU.
- Resource Manager: Manage global resources such as random number generator, temporal space.
- NDArray: Dynamic asynchronize n-dimensional arrays, provide flexible
  imperative programs for MXNet.
- Symbolic Execution: Static symbolic graph executor, provide efficient symbolic
  graph execution and optimization.
- [Operator](operator.md): Operators that defines static forward and gradient
  calculation(backprop).
- [SimpleOp](operator_util.md): Operators that extend to NDArray operators and symbolic operators
  in a unified fashion.
- Symbol Construction: Symbolic construction, provide a way to construct
  computation graph(net configuration)
- [KVStore](multi_node.md): Key-value store interface for easy parameter synchronizations.
- Data Loading(IO): Efficient distributed data loading and augmentation.


## How to Read the Code
- All the module interface are listed in [include](../../include), these
  interfaces are heavily documented.
- You read the
  [Doxygen Version](https://mxnet.readthedocs.org/en/latest/doxygen) of the
  document.
- Each module will only depend on other module by the header files in
  [include](../../include).
- The implementation of module is in [src](../../src) folder.
- Each source code only sees the file within its folder,
  [src/common](../../src/common) and [include](../../include).

Most modules are mostly self-contained, with interface dependency on engine.  So
you are free to pick the one you are interested in, and read that part.

### Analogy to CXXNet
- The Symbolic Execution can be viewed as neural net execution(forward,
  backprop) with more optimizations.
- The Operator can be viewed as Layers, but need to pass in weights and bias.
	- It also contains more(optional) interface to further optimize memory usage.
- The Symbolic Construction module is advanced config file.
- The Runtime Dependency Engine engine is like a thread pool.
	- But makes your life easy to solve dependency tracking for you.
- KVStore adopts a simple parameter-server interface optimized for GPU
  synchronization.

### Analogy to Minerva
- The Runtime Dependency Engine is DAGEngine in Minerva, except that it is
  enhanced to support mutations.
- The NDArray is same as owl.NDArray, except that it supports mutation, and can
  interact with Symbolic Execution.

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
