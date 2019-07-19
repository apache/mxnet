# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


#############################################################
## Please read the README.md document for better reference ##
#############################################################
from __future__ import print_function

import logging
import random

import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

np.random.seed(1234) # set seed for deterministic ordering
mx.random.seed(1234)
random.seed(1234)

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
mlp_svm_l2 = mx.symbol.SVMOutput(data=fc3, name='svm_l2')

# With L1-SVM objective
mlp_svm_l1 = mx.symbol.SVMOutput(data=fc3, name='svm_l1', use_linear=True)

# Compare with softmax cross entropy loss
mlp_softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

print("Preparing data...")
mnist_data = mx.test_utils.get_mnist()
X = np.concatenate([mnist_data['train_data'], mnist_data['test_data']])
Y = np.concatenate([mnist_data['train_label'], mnist_data['test_label']])
X = X.reshape((X.shape[0], -1)).astype(np.float32) * 255

# Now we fetch MNIST dataset, add some noise, as the article suggests,
# permutate and assign the examples to be used on our network
mnist_pca = PCA(n_components=70).fit_transform(X)
noise = np.random.normal(size=mnist_pca.shape)
mnist_pca += noise
p = np.random.permutation(mnist_pca.shape[0])
X = mnist_pca[p] / 255.
Y = Y[p]
X_show = X[p]

# This is just to normalize the input and separate train set and test set
X_train = X[:60000]
X_test = X[60000:]
X_show = X_show[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]
print("Data prepared.")
# Article's suggestion on batch size
batch_size = 200

ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

results = {}
for output in [mlp_svm_l2, mlp_svm_l1, mlp_softmax]:
    
    print("\nTesting with %s \n" % output.name)
    
    label = output.name + "_label"
    
    train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size, label_name=label)
    test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size, label_name=label)

    # Here we instatiate and fit the model for our data
    # The article actually suggests using 400 epochs,
    # But I reduced to 10, for convenience

    mod = mx.mod.Module(
        context = ctx, 
        symbol = output,         # Use the network we just defined
        label_names = [label],
    )
    mod.fit(
        train_data=train_iter,
        eval_data=test_iter,  # Testing data set. MXNet computes scores on test set every epoch
        batch_end_callback = mx.callback.Speedometer(batch_size, 200),  # Logging module to print out progress
        num_epoch = 10,       # Train for 10 epochs
        optimizer_params = {
            'learning_rate': 0.1,  # Learning rate
            'momentum': 0.9,       # Momentum for SGD with momentum
            'wd': 0.00001,         # Weight decay for regularization
        })
    results[output.name] = mod.score(test_iter, mx.metric.Accuracy())[0][1]*100
    print('Accuracy for %s:'%output.name, mod.score(test_iter, mx.metric.Accuracy())[0][1]*100, '%\n')
    
for key, value in results.items():
    print(key, value, "%s")

#svm_l2 97.85 %s
#svm_l1 98.15 %s
#softmax 97.69 %s
