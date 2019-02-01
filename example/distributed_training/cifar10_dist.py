#!/usr/bin/env python

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

"""cifar10_dist.py contains code that trains a ResNet18 network using distributed training"""

from __future__ import print_function

import sys
import random
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon, kv, nd
from mxnet.gluon.model_zoo import vision

# Create a distributed key-value store
store = kv.create('dist')

# Clasify the images into one of the 10 digits
num_outputs = 10

# 64 images in a batch
batch_size_per_gpu = 64
# How many epochs to run the training
epochs = 5

# How many GPUs per machine
gpus_per_machine = 4
# Effective batch size across all GPUs
batch_size = batch_size_per_gpu * gpus_per_machine

# Create the context (a list of all GPUs to be used for training)
ctx = [mx.gpu(i) for i in range(gpus_per_machine)]


# Convert to float 32
# Having channel as the first dimension makes computation more efficient. Hence the (2,0,1) transpose.
# Dividing by 255 normalizes the input between 0 and 1
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1))/255, label.astype(np.float32)


class SplitSampler(gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with index `part_index`

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len

    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len


# Load the training data
train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True, transform=transform), batch_size,
                                   sampler=SplitSampler(50000, store.num_workers, store.rank))

# Load the test data
test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False, transform=transform),
                                  batch_size, shuffle=False)

# Use ResNet from model zoo
net = vision.resnet18_v1()

# Initialize the parameters with Xavier initializer
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# SoftmaxCrossEntropy is the most common choice of loss function for multiclass classification
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# Use Adam optimizer. Ask trainer to use the distributor kv store.
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001}, kvstore=store)


# Evaluate accuracy of the given network using the given data
def evaluate_accuracy(data_iterator, network):
    """ Measure the accuracy of ResNet

    Parameters
    ----------
    data_iterator: Iter
      examples of dataset
    network:
      ResNet

    Returns
    ----------
    tuple of array element
    """
    acc = mx.metric.Accuracy()

    # Iterate through data and label
    for i, (data, label) in enumerate(data_iterator):

        # Get the data and label into the GPU
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])

        # Get network's output which is a probability distribution
        # Apply argmax on the probability distribution to get network's classification.
        output = network(data)
        predictions = nd.argmax(output, axis=1)

        # Give network's prediction and the correct label to update the metric
        acc.update(preds=predictions, labels=label)

    # Return the accuracy
    return acc.get()[1]


# We'll use cross entropy loss since we are doing multiclass classification
loss = gluon.loss.SoftmaxCrossEntropyLoss()


# Run one forward and backward pass on multiple GPUs
def forward_backward(network, data, label):

    # Ask autograd to remember the forward pass
    with autograd.record():
        # Compute the loss on all GPUs
        losses = [loss(network(X), Y) for X, Y in zip(data, label)]

    # Run the backward pass (calculate gradients) on all GPUs
    for l in losses:
        l.backward()


# Train a batch using multiple GPUs
def train_batch(batch_list, context, network, gluon_trainer):
    """ Training with multiple GPUs

    Parameters
    ----------
    batch_list: List
      list of dataset
    context: List
      a list of all GPUs to be used for training
    network:
      ResNet
    gluon_trainer:
      rain module of gluon
    """
    # Split and load data into multiple GPUs
    data = batch_list[0]
    data = gluon.utils.split_and_load(data, context)

    # Split and load label into multiple GPUs
    label = batch_list[1]
    label = gluon.utils.split_and_load(label, context)

    # Run the forward and backward pass
    forward_backward(network, data, label)

    # Update the parameters
    this_batch_size = batch_list[0].shape[0]
    gluon_trainer.step(this_batch_size)


# Run as many epochs as required
for epoch in range(epochs):

    # Iterate through batches and run training using multiple GPUs
    batch_num = 1
    for batch in train_data:

        # Train the batch using multiple GPUs
        train_batch(batch, ctx, net, trainer)

        batch_num += 1

    # Print test accuracy after every epoch
    test_accuracy = evaluate_accuracy(test_data, net)
    print("Epoch %d: Test_acc %f" % (epoch, test_accuracy))
    sys.stdout.flush()
