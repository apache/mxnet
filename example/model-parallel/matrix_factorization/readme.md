Model Parallel Matrix Factorization
==============

The example demonstrates the basic usage of `group2ctxs` in `Module`, which allows one part of the model trained on cpu
and the other on gpu.

- `python train.py --num-gpus 2`

This example also walks you through a matrix factorization algorithm for recommendations. It is applied to a dataset of
10 million movie ratings available from [the MovieLens project](http://files.grouplens.org/datasets/movielens/).
