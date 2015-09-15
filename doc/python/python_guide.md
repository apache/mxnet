# MXNet Python Guide

This page gives a general overvie of MXNet python package. MXNet contains a
mixed flavor of elements you might need to bake flexible and efficient
applications. There are mainly three concepts in MXNet:

* Numpy style `NDArray` offers matrix and tensor computations on both CPU and
GPU, with automatic parallelization

* `Symbol` makes defining a neural network extremely easy, and it provides
  automatic differentiation.

* `KVStore` allows data synchronization between multi-GPUs and multi-machine
  easily

**Table of contents**

```eval_rst
.. toctree::
   :maxdepth: 2

   ndarray
   symbol
   kvstore
   io
```



<!-- How to Choose between APIs -->
<!-- -------------------------- -->
<!-- You can mix them all as much as you like. Here are some guidelines -->
<!-- * Use Symbolic API and coarse grained operator to create established structure. -->
<!-- * Use fine-grained operator to extend parts of of more flexible symbolic graph. -->
<!-- * Do some dynamic NArray tricks, which are even more flexible, between the calls of forward and backward of executors. -->

<!-- We believe that different ways offers you different levels of flexibilty and efficiency. Normally you do not need to -->
<!-- be flexible in all parts of the networks, so we allow you to use the fast optimized parts, -->
<!-- and compose it flexibly with fine-grained operator or dynamic NArray. We believe such kind of mixture allows you to build -->
<!-- the deep learning architecture both efficiently and flexibly as your choice. To mix is to maximize the peformance and flexiblity. -->
