Factorization Machine
===========
This example trains a factorization machine model using the criteo dataset.

## Download the Dataset

The criteo dataset is available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#criteo
The data was used in a competition on click-through rate prediction jointly hosted by Criteo and Kaggle in 2014,
with 1,000,000 features. There are 45,840,617 training examples and 6,042,135 testing examples.
It takes more than 30 GB to download and extract the dataset.

## Train the Model

- python train.py --train-data /path/to/criteo.kaggle2014.test.svm --test-data /path/to/criteo.kaggle2014.test.svm

[Rendle, Steffen. "Factorization machines." In Data Mining (ICDM), 2010 IEEE 10th International Conference on, pp. 995-1000. IEEE, 2010. ](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
