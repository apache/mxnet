Linear Classification Using Sparse Matrix Multiplication
===========
This examples trains a linear model using the sparse feature in MXNet. This is for demonstration purpose only.

The example utilizes the sparse data loader ([mx.io.LibSVMIter](https://mxnet.incubator.apache.org/versions/master/api/python/io/io.html#mxnet.io.LibSVMIter)),
the sparse dot operator and [sparse gradient updaters](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/sparse.html#updater)
to train a linear model on the
[Avazu](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#avazu) click-through-prediction dataset.

The example also shows how to perform distributed training with the sparse feature.

- `python train.py`

Notes on Distributed Training:

- For distributed training, please use the `../../tools/launch.py` script to launch a cluster.
- For example, to run two workers and two servers with one machine, run `../../../tools/launch.py -n 2 --launcher=local python train.py --kvstore=dist_async`
