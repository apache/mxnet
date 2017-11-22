Factorization Machine
===========
This example trains a factorization machine model using the criteo dataset.

## Download the Dataset

The provided dataset is a pre-processed [criteo dataset from the kaggle challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
in the [LibSVM format](https://mxnet.incubator.apache.org/versions/master/api/python/io.html#mxnet.io.LibSVMIter)
in MXNet, whose features are re-hashed to 2 million. The total size of the dataset is around 13 GB.

- python data.py --dir /path/to/criteo/folder/

## Train the Model

- python train.py --data /path/to/criteo/folder/

[Rendle, Steffen. "Factorization machines." In Data Mining (ICDM), 2010 IEEE 10th International Conference on, pp. 995-1000. IEEE, 2010. ](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
