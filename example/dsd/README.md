DSD Training
============
This folder contains an optimizer class that implements DSD training coupled with SGD. The training
procedure is described in the paper *DSD: Dense-Sparse-Dense Training for Deep Neural Networks*,
available at https://arxiv.org/pdf/1607.04381.pdf

The optimizer class is flexible in the way it prunes weights. The user can define the following:
-   The percentage sparsity they want or the threshlding value for the pruning
-   The epochs at which they want a particular level of pruning
