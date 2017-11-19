# Tutorials

These tutorials introduce a few fundamental concepts in deep learning and how to implement them in _MXNet_. The _Basics_ section contains tutorials on manipulating arrays, building networks, loading/preprocessing data, etc. The _Training and Inference_ section talks about implementing Linear Regression, training a Handwritten digit classifier using MLP and CNN, running inferences using a pre-trained model, and lastly, efficiently training a large scale image classifier.


## Gluon

Gluon is the high-level interface for MXNet. It is more intuitive and easier to use than the lower level interface.
Gluon supports dynamic (define-by-run) graphs with JIT-compilation to achieve both flexibility and efficiency.
This is a selected subset of Gluon tutorials. For the comprehensive tutorial on Gluon,
please see [gluon.mxnet.io](http://gluon.mxnet.io).

### Basics

- [Manipulate data the MXNet way with ndarray](http://gluon.mxnet.io/chapter01_crashcourse/ndarray.html)
- [Automatic differentiation with autograd](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html)
- [Linear regression with gluon](http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html)
- [Serialization - saving, loading and checkpointing](http://gluon.mxnet.io/chapter03_deep-neural-networks/serialization.html)

### Neural Networks

- [Multilayer perceptrons in gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html)
- [Convolutional Neural Networks in gluon](http://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-gluon.html)
- [Recurrent Neural Networks with gluon](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/rnns-gluon.html)

### Advanced

- [Plumbing: A look under the hood of gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html)
- [Designing a custom layer with gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html)
- [Fast, portable neural networks with Gluon HybridBlocks](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)
- [Training on multiple GPUs with gluon](http://gluon.mxnet.io/chapter07_distributed-learning/multiple-gpus-gluon.html)

## MXNet

### Basics

```eval_rst
.. toctree::
   :maxdepth: 1

   basic/ndarray
   basic/symbol
   basic/module
   basic/data
```

### Training and Inference

```eval_rst
.. toctree::
   :maxdepth: 1

   python/linear-regression
   python/mnist
   python/predict_image
   vision/large_scale_classification
```

### Sparse NDArray

```eval_rst
.. toctree::
   :maxdepth: 1

   sparse/csr
   sparse/row_sparse
   sparse/train
```

<br>
More tutorials and examples are available in the GitHub [repository](https://github.com/dmlc/mxnet/tree/master/example).

## Contributing Tutorials

Want to contribute an MXNet tutorial? To get started, download the [tutorial template](https://github.com/dmlc/mxnet/tree/master/example/MXNetTutorialTemplate.ipynb).
