# Tutorials

## NDArray

NDArray is MXNet’s primary tool for storing and transforming data. NDArrays are similar to NumPy's multi-dimensional array. However, they confer a few key advantages. First, NDArrays support asynchronous computation on CPU, GPU, and distributed cloud architectures. Second, they provide support for automatic differentiation. These properties make NDArray an ideal library for machine learning, both for researchers and engineers launching production systems.

- [Manipulate data the MXNet way with ndarray](http://gluon.mxnet.io/chapter01_crashcourse/ndarray.html)


## Automatic gradients

MXNet makes it easier to calculate derivatives by automatically calculating them while writing ordinary imperative code. Every time you a make pass through your model, autograd builds a graph on the fly, through which it can immediately backpropagate gradients.

- [Automatic differentiation with autograd](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html)


## Gluon

Gluon is MXNet's imperative API. It is more intuitive and easier to use than the symbolic API. Gluon supports dynamic (define-by-run) graphs with JIT-compilation to achieve both flexibility and efficiency.

This is a selected subset of Gluon tutorials that explains basic usage of Gluon and fundamental concepts in deep learning. For the comprehensive tutorial on Gluon that covers topics from basic statistics and probability theory to reinforcement learning and recommender systems, please see gluon.mxnet.io.

### Basics

- [Linear regression with gluon](http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html)
- [Serialization - saving, loading and checkpointing](http://gluon.mxnet.io/chapter03_deep-neural-networks/serialization.html)

### Neural Networks

- [Multilayer perceptrons in gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html)
- [Convolutional Neural Networks in gluon](http://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-gluon.html)
- [Recurrent Neural Networks with gluon](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/rnns-gluon.html)

### Advanced

- [Plumbing: A look under the hood of gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html)
- [Designing a custom layer with gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html)
- [Training on multiple GPUs with gluon](http://gluon.mxnet.io/chapter07_distributed-learning/multiple-gpus-gluon.html)


## Symbolic Interface

MXNet's symbolic interface lets users define a computation graph first and then execute it using MXNet. This enables MXNet to perform a lot of optimizations that are not possible in imperative execution (like operator folding and safe reuse of memory used by temporary variables).


```eval_rst
.. toctree::
   :maxdepth: 1

   basic/symbol
   basic/module
   basic/data
   python/mnist
   python/predict_image
```


## Hybrid Networks

Imperative programs are very intuitive to write and are very flexible. But symbolic programs tend to be more efficient. MXNet combines both these paradigms to give users the best of both worlds. Users can write intuitive imperative code during development and MXNet will automatically generate symbolic execution graph for faster execution.

- [Fast, portable neural networks with Gluon HybridBlocks](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)

## Sparse operations

A lot of real-world datasets are very sparse (very few nonzeros). MXNet's sparse operations help store these sparse matrices is a memory efficient way and perform computations on them much faster.

```eval_rst
.. toctree::
   :maxdepth: 1

   sparse/csr
   sparse/row_sparse
   sparse/train
```

## Performance

A lot of real-world datasets are too huge to train models on a single GPU or a single machine. MXNet solves this problem by scaling almost linearly across multiple GPUs and multiple machines.

```eval_rst
.. toctree::
   :maxdepth: 1

   vision/large_scale_classification
```


<br>
More tutorials and examples are available in the GitHub [repository](https://github.com/apache/incubator-mxnet/tree/master/example).


## Contributing Tutorials

Want to contribute an MXNet tutorial? To get started, download the [tutorial template](https://github.com/apache/incubator-mxnet/tree/master/example/MXNetTutorialTemplate.ipynb).
