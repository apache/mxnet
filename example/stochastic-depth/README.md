Stochastic Depth
================

This folder contains examples showing implementation of the stochastic depth algorithm described in the paper
Huang, Gao, et al. ["Deep networks with stochastic depth."](https://arxiv.org/abs/1603.09382)
arXiv preprint arXiv:1603.09382 (2016). This paper introduces a new way to perturb networks during training
in order to improve their performance. Stochastic Depth (SD) is a method for residual networks,
which randomly removes/deactivates residual blocks during training.

The paper talks about constructing the network of residual blocks which are basically a set of
convolution layers and a bypass that passes the information from the previous layer through without any change.
With stochastic depth, the convolution block is sometimes switched off allowing the information
to flow through the layer without being changed, effectively removing the layer from the network.
During testing, all layers are left in and the weights are modified by their survival probability.
This is very similar to how dropout works, except instead of dropping a single node in a layer
the entire layer is dropped!

The main idea behind stochastic depth is relatively simple, but the results are surprisingly good.
The authors demonstrated the new architecture on CIFAR-10, CIFAR-100, and the Street View House Number dataset (SVHN).
They achieve the lowest published error on CIFAR-10 and CIFAR-100, and second lowest for SVHN.

Files in this example folder:

- `sd_mnist.py` example shows sample implementation of the algorithm just for the sanity check.

- **sd_cifar10.py** shows the algorithm implementation for 500 epochs on cifar_10 dataset. After 500 epochs, ~9.4% error
was achieved for cifar10, it can be further improved by some more careful hyper parameters tuning to achieve
the reported numbers in the paper.
You can see the sample result log in the top section of sd_cifar10.py file.
