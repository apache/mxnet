Matrix Factorization w/ Sparse Embedding
===========
The example demonstrates the basic usage of the sparse.Embedding operator in MXNet, adapted based on @leopd's recommender examples.
This is for demonstration purpose only.

```
usage: train.py [-h] [--num-epoch NUM_EPOCH] [--seed SEED]
                [--batch-size BATCH_SIZE] [--log-interval LOG_INTERVAL]
                [--factor-size FACTOR_SIZE] [--gpus GPUS] [--dense]

Run matrix factorization with sparse embedding

optional arguments:
  -h, --help            show this help message and exit
  --num-epoch NUM_EPOCH
                        number of epochs to train (default: 3)
  --seed SEED           random seed (default: 1)
  --batch-size BATCH_SIZE
                        number of examples per batch (default: 128)
  --log-interval LOG_INTERVAL
                        logging interval (default: 100)
  --factor-size FACTOR_SIZE
                        the factor size of the embedding operation (default: 128)
  --gpus GPUS           list of gpus to run, e.g. 0 or 0,2. empty means using
                        cpu(). (default: None)
  --dense               whether to use dense embedding (default: False)
```
