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


# An example to train a deep learning model with differential privacy
# Author: Yu-Xiang Wang


# import packages for DP
from pydiffpriv import cgfbank, dpacct

# import packages needed for deep learning
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import dpdl_utils

ctx = mx.cpu()


# ## Get data:  standard MNIST


mnist = mx.test_utils.get_mnist()
num_inputs = 784
num_outputs = 10
batch_size = 1 # this is set to get per-example gradient



train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"],
                               batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"],
                              64, shuffle=True)
train_data2 = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"],
                               64, shuffle=True)


# ## Build a one hidden layer NN with Gluon



num_hidden = 1000
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, in_units=num_inputs,activation="relu"))
    net.add(gluon.nn.Dense(num_outputs,in_units=num_hidden))

# get and save the parameters
params = net.collect_params()
params.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
params.setattr('grad_req', 'write')

# define loss function
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


# ## Use a new optimizer called privateSGD
# Basically, we add Gaussian noise to the stochastic gradient.


# define the update rule
def privateSGD(x, g, lr, sigma,wd=0.0,ctx=mx.cpu()):
    for (param,grad) in zip(x.values(), g):
        v=param.data()
        v[:] = v - lr * (grad +wd*v+ sigma*nd.random_normal(shape = grad.shape).as_in_context(ctx))
# Utility function to evaluate error

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    loss_fun = .0
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx).reshape((-1, 784))
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        loss = softmax_cross_entropy(output, label)
        loss_fun = loss_fun*i/(i+1) + nd.mean(loss).asscalar()/(i+1)
    return acc.get()[1], loss_fun


# ## Now let's try attaching a privacy accountant to this data set



# declare a moment accountant from pydiffpriv
DPobject = dpacct.anaCGFAcct()

# Specify privacy specific inputs
thresh = 4.0 # limit the norm of individual gradient
sigma = thresh

delta = 1e-5

func = lambda x: cgfbank.CGF_gaussian({'sigma': sigma/thresh}, x)


# ## We now specify the parameters needed for learning

#
epochs = 10
learning_rate = .1

n = train_data.num_data
batchsz = 100 #

count = 0
niter=0
moving_loss = 0

grads = dpdl_utils.initialize_grad(params,ctx=ctx)


# ## Let's start then!


# declare a few place holder for logging
logs = {}
logs['eps'] = []
logs['loss'] = []
logs['MAloss'] = []
logs['train_acc'] = []
logs['test_acc'] = []




for e in range(epochs):
    # train_data.reset()  # Reset does not shuffle yet
    train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"],
                                   batch_size, shuffle=True)
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx).reshape((-1, 784))
        label = batch.label[0].as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()

        # calculate an moving average estimate of the loss
        count += 1
        moving_loss = .999 * moving_loss + .001 * nd.mean(loss).asscalar()
        est_loss = moving_loss / (1 - 0.999 ** count)

        # Add up the clipped individual gradient
        dpdl_utils.accumuate_grad(grads, params, thresh)

        #print(i)
        if not (i + 1) % batchsz:  # update the parameters when we collect enough data

            privateSGD(params, grads, learning_rate/batchsz,sigma,wd=0.1,ctx=ctx)

            # Keep track of the privacy loss
            DPobject.compose_subsampled_mechanism(func,1.0*batchsz/n)


            dpdl_utils.reset_grad(grads)

        if count % (10*batchsz) is 0:
            print("[%s] Loss: %s. Privacy loss: eps = %s, delta = %s " % (((count+1)/batchsz),est_loss,DPobject.get_eps(delta),delta))
            logs['MAloss'].append(est_loss)
        ##########################
        #  Keep a moving average of the losses
        ##########################

        if count % 60000 is 0:
            test_accuracy, loss_test = evaluate_accuracy(test_data, net)
            train_accuracy, loss_train = evaluate_accuracy(train_data2, net)

            print("Net: Epoch %s. Train Loss: %s, Test Loss: %s, Train_acc %s, Test_acc %s" %
                 (e, loss_train, loss_test,train_accuracy, test_accuracy))

            logs['eps'].append(DPobject.get_eps(delta))
            logs['loss'].append(loss_train)
            logs['train_acc'].append(train_accuracy)
            logs['test_acc'].append(test_accuracy)

            learning_rate = learning_rate/2



## Plot some figures!


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(range(epochs), logs['eps'])
plt.plot(range(epochs), logs['loss'])
plt.plot(range(epochs), logs['train_acc'])
plt.plot(range(epochs), logs['test_acc'])

plt.legend(['\delta = 1e-5', 'Training loss', 'Training accuracy','Test accuracy'], loc='best')
plt.show()

