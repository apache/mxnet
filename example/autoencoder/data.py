import numpy as np
from sklearn.datasets import fetch_mldata

def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    mnist = fetch_mldata('MNIST original', data_home='../../data')
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p].astype(np.float32)*0.02
    Y = mnist.target[p]
    return X, Y
