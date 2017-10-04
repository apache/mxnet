Example
===========
This folder contains examples using the sparse feature in MXNet.

## Linear Classification

The example demonstrates the basic usage of the sparse feature in MXNet to speedup computation. It utilizes the sparse data loader, sparse operators and a sparse gradient updater to train a linear model on the [Avazu](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#avazu) click-through-prediction dataset.

- `python linear_classification.py`

Notes on Distributed Training:

- For distributed training, please use the `../../tools/launch.py` script to launch a cluster.
- For example, to run two workers and two servers with one machine, run `../../tools/launch.py -n 2 --launcher=local python linear_classification.py --kvstore=dist_async`
