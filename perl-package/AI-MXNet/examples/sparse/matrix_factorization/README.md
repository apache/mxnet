Matrix Factorization w/ Sparse Embedding
===========
The example demonstrates the basic usage of the SparseEmbedding operator in MXNet, adapted based on @leopd's recommender examples.
The operator is available on both CPU and GPU. This is for demonstration purpose only.

- get_data.sh
- perl train.pl
- To compare the training speed with (dense) Embedding, run perl train.pl --use-dense 1
- To run the example on the GPU, run perl train.pl --use-gpu 1
