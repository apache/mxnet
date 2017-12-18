Model Parallel Matrix Factorization
===================================

This example walks you through a matrix factorization algorithm for recommendations and also
demonstrates the basic usage of `group2ctxs` in `Module`, which allows one part of the model to be
trained on cpu and the other on gpu. So, it is necessary to have GPUs available on the machine
to run this example.

To run this example, first make sure you download a dataset of 10 million movie ratings available
from [the MovieLens project](http://files.grouplens.org/datasets/movielens/) by running following command:

`python get_data.py`

This will download MovieLens 10M dataset under ml-10M100K folder. Now, you can run the training as follows:

`python train.py --num-gpus 1`

You can also specify other attributes such as num-epoch, batch-size,
factor-size(output dim of the embedding operation) to train.py.

While training you will be able to see the usage of ctx_group attribute to divide the operators
into different groups corresponding to different CPU/GPU devices.
