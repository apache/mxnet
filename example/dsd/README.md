DSD Training
============
This folder contains an optimizer class that implements DSD training coupled with SGD. The training
procedure is described in the paper *DSD: Dense-Sparse-Dense Training for Deep Neural Networks*,
available at https://arxiv.org/pdf/1607.04381.pdf

The optimizer class is flexible in the way it prunes weights. The user can define the following:
-   The percentage sparsity they want or the thresholding value for the pruning
-   The epochs at which they want a particular level of pruning

Note that giving the sparsity level induces that level of sparsity in every layer of the neural
network. It layer-wise pruning, and not global pruning (which would require loooking at all the
weights of the neural network at the same time). However, global pruning can be done if the
threshold value is known to the user (by doing some preprocessing), and is passed to the optimizer.
