# MXNet Documentation

[MXNet.jl](https://github.com/dmlc/MXNet.jl) is the
[Julia](http://julialang.org/) package of
[dmlc/mxnet](https://github.com/dmlc/mxnet). MXNet.jl brings flexible and efficient GPU
computing and state-of-art deep learning to Julia. Some highlight of features
include:

* Efficient tensor/matrix computation across multiple devices,
  including multiple CPUs, GPUs and distributed server nodes.
* Flexible symbolic manipulation to composite and construct
  state-of-the-art deep learning models.

For more details, see documentation below. Please also checkout the
[examples](https://github.com/dmlc/MXNet.jl/tree/master/examples) directory.

## Tutorials

```@contents
Pages = [
  "tutorial/mnist.md",
  "tutorial/char-lstm.md",
]
Depth = 2
```

## User's Guide

```@contents
Pages = [
  "user-guide/install.md",
  "user-guide/overview.md",
  "user-guide/faq.md",
]
Depth = 2
```

## API Documentation

```@contents
Pages = [
  "api/context.md",
  "api/ndarray.md",
  "api/symbolic-node.md",
  "api/model.md",
  "api/initializers.md",
  "api/optimizers.md",
  "api/callbacks.md",
  "api/metric.md",
  "api/io.md",
  "api/nn-factory.md",
  "api/executor.md",
  "api/visualize.md",
]
```
