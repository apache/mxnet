Stochastic depth
================

This folder contains examples showing implementation of the stochastic depth algorithm described in the paper
Huang, Gao, et al. "Deep networks with stochastic depth." arXiv preprint arXiv:1603.09382 (2016).
This paper introduces a new way to perturb networks during training in order to improve their performance.

**sd_mnist.py** example shows sample implementation of the algorithm just for the sanity check.

**sd_cifar10.py** shows the algorithm implementation for 500 epochs on cifar_10 dataset. After 500 epochs, ~9.4% error
was achieved for cifar10, it can be further improved by some more careful hyper parameters tuning to achieve
the reported numbers in the paper.