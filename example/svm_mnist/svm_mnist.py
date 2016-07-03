
#############################################################
## Please read the README.md document for better reference ##
#############################################################

import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Network declaration as symbols. The following pattern was based
# on the article, but feel free to play with the number of nodes
# and with the activation function
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=512)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 512)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)

# Here we add the ultimate layer based on L2-SVM objective
mlp = mx.symbol.SVMOutput(data=fc3, name='svm')

# To use L1-SVM objective, comment the line above and uncomment the line below
# mlp = mx.symbol.SVMOutput(data=fc3, name='svm', use_linear=True)

# Now we fetch MNIST dataset, add some noise, as the article suggests,
# permutate and assign the examples to be used on our network
mnist = fetch_mldata('MNIST original')
mnist_pca = PCA(n_components=70).fit_transform(mnist.data)
noise = np.random.normal(size=mnist_pca.shape)
mnist_pca += noise
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist_pca.shape[0])
X = mnist_pca[p]
Y = mnist.target[p]
X_show = mnist.data[p]

# This is just to normalize the input to a value inside [0,1],
# and separate train set and test set
X = X.astype(np.float32)/255
X_train = X[:60000]
X_test = X[60000:]
X_show = X_show[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

# Article's suggestion on batch size
batch_size = 200
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)

# A quick work around to prevent mxnet complaining the lack of a softmax_label
train_iter.label =  mx.io._init_data(Y_train, allow_empty=True, default_name='svm_label')
test_iter.label =  mx.io._init_data(Y_test, allow_empty=True, default_name='svm_label')

# Here we instatiate and fit the model for our data
# The article actually suggests using 400 epochs,
# But I reduced to 10, for convinience
model = mx.model.FeedForward(
    ctx = mx.cpu(0),      # Run on CPU 0
    symbol = mlp,         # Use the network we just defined
    num_epoch = 10,       # Train for 10 epochs
    learning_rate = 0.1,  # Learning rate
    momentum = 0.9,       # Momentum for SGD with momentum
    wd = 0.00001,         # Weight decay for regularization
    )
model.fit(
    X=train_iter,  # Training data set
    eval_data=test_iter,  # Testing data set. MXNet computes scores on test set every epoch
    batch_end_callback = mx.callback.Speedometer(batch_size, 200))  # Logging module to print out progress

# Uncomment to view an example
# plt.imshow((X_show[0].reshape((28,28))*255).astype(np.uint8), cmap='Greys_r')
# plt.show()
# print 'Result:', model.predict(X_test[0:1])[0].argmax()

# Now it prints how good did the network did for this configuration
print 'Accuracy:', model.score(test_iter)*100, '%'