from __future__ import print_function
import numpy
import os
import ssl


def load_mnist(training_num=50000):
    data_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'mnist.npz')
    if not os.path.isfile(data_path):
        from six.moves import urllib
        origin = (
            'https://github.com/sxjscience/mxnet/raw/master/example/bayesian-methods/mnist.npz'
        )
        print('Downloading data from %s to %s' % (origin, data_path))
        context = ssl._create_unverified_context()
        urllib.request.urlretrieve(origin, data_path, context=context)
        print('Done!')
    dat = numpy.load(data_path)
    X = (dat['X'][:training_num] / 126.0).astype('float32')
    Y = dat['Y'][:training_num]
    X_test = (dat['X_test'] / 126.0).astype('float32')
    Y_test = dat['Y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test


def load_toy():
    training_data = numpy.loadtxt('toy_data_train.txt')
    testing_data = numpy.loadtxt('toy_data_test_whole.txt')
    X = training_data[:, 0].reshape((training_data.shape[0], 1))
    Y = training_data[:, 1].reshape((training_data.shape[0], 1))
    X_test = testing_data[:, 0].reshape((testing_data.shape[0], 1))
    Y_test = testing_data[:, 1].reshape((testing_data.shape[0], 1))
    return X, Y, X_test, Y_test


def load_synthetic(theta1, theta2, sigmax, num=20):
    flag = numpy.random.randint(0, 2, (num,))
    X = flag * numpy.random.normal(theta1, sigmax, (num,)) \
        + (1.0 - flag) * numpy.random.normal(theta1 + theta2, sigmax, (num,))
    return X
