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

## Example

To test out the sparsity feature on a MLP, run the following script:

    python mlp.py --pruning_switch_epoch 4,7,10 --bias_sparsity 0,30,50 --weight_sparsity 0,50,70

This will train a MLP with 0% sparsity uptil epoch 4, with 30% bias and 50% weight sparsity uptil
epoch 7, 50% bias and 70% weight sparsity uptil epoch 10.

To test out the thresholding feature on a MLP, run the following script:

    python mlp.py --pruning_switch_epoch 4,6 --bias_threshold 0,0.01 --weight_threshold 0,0.05

This will train a MLP with thresholding at 0 uptil epoch 4, with bias thresholding at 0.01 and
weight thresholding at 0.05 uptil epoch 6.
