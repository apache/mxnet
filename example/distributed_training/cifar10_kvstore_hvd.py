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

"""cifar10_dist_hvd.py contains code that runs distributed training of a
ResNet18 network using Horovod framework"""

import argparse
import logging
import time
import random
import types
import warnings

import numpy as np
import mxnet as mx
from mxnet import autograd, gluon, kv, nd
from mxnet.gluon.model_zoo import vision

logging.basicConfig(level=logging.INFO)

# Training settings
parser = argparse.ArgumentParser(description='MXNet CIFAR Example')

parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size per worker (default: 64)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable training on GPU (default: False)')
args = parser.parse_args()

if not args.no_cuda:
    # Disable CUDA if there are no GPUs.
    if mx.device.num_gpus() == 0:
        args.no_cuda = True


# Transform input data
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1))/255,\
      label.astype(np.float32)


# Train a batch using multiple GPUs
def train(batch_list, context, network, gluon_trainer, metric):
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

    # Run one forward and backward pass
    def forward_backward(network, data, labels, metric):
        with autograd.record():
            # Compute outputs
            outputs = [network(X) for X in data]
            # Compute the loss
            losses = [loss(yhat, y) for yhat, y in zip(outputs, labels)]

        # Run the backward pass (calculate gradients)
        for l in losses:
            l.backward()

        metric.update(preds=outputs, labels=labels)

    # Use cross entropy loss
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # Split and load data
    data = batch_list[0]
    data = gluon.utils.split_and_load(data, context)

    # Split and load label
    label = batch_list[1]
    label = gluon.utils.split_and_load(label, context)

    # Run the forward and backward pass
    forward_backward(network, data, label, metric)

    # Update the parameters
    this_batch_size = batch_list[0].shape[0]
    gluon_trainer.step(this_batch_size)


# Evaluate accuracy of the given network using the given data
def evaluate(data_iterator, network, context):
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
    acc = mx.gluon.metric.Accuracy()

    # Iterate through data and label
    for i, (data, label) in enumerate(data_iterator):

        # Get the data and label into the GPU
        data = data.as_in_context(context)
        label = label.as_in_context(context)

        # Get network's output which is a probability distribution
        # Apply argmax on the probability distribution to get network's
        # classification.
        output = network(data)
        predictions = nd.argmax(output, axis=1)

        # Give network's prediction and the correct label to update the metric
        acc.update(preds=predictions, labels=label)

    # Return the accuracy
    return acc.get()[1]


class SplitSampler(gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with
    index `part_index`

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


# Use Horovod as the KVStore
store = kv.create('horovod')

# Get the number of workers
num_workers = store.num_workers

# Create the context based on the local rank of the current process
ctx = mx.cpu(store.local_rank) if args.no_cuda else mx.gpu(store.local_rank)

# Load the training data
train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True,
                                   transform=transform), args.batch_size,
                                   sampler=SplitSampler(50000,
                                                        num_workers,
                                                        store.rank))

# Load the test data
test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False,
                                  transform=transform),
                                  args.batch_size, shuffle=False)

# Load ResNet18 model from GluonCV model zoo
net = vision.resnet18_v1()

# Initialize the parameters with Xavier initializer
net.initialize(mx.init.Xavier(), ctx=ctx)

# Use Adam optimizer. Ask trainer to use the distributor kv store.
trainer = gluon.Trainer(net.collect_params(), optimizer='adam',
                        optimizer_params={'learning_rate': args.lr},
                        kvstore=store)

train_metric = mx.gluon.metric.Accuracy()

# Run as many epochs as required
for epoch in range(args.epochs):
    tic = time.time()
    train_metric.reset()

    # Iterate through batches and run training using multiple GPUs
    batch_num = 1
    btic = time.time()
    for batch in train_data:
        # Train the batch using multiple GPUs
        train(batch, [ctx], net, trainer, train_metric)
        if store.rank == 0 and batch_num % 100 == 0:
            speed = args.batch_size / (time.time() - btic)
            logging.info('Epoch[{}] Rank [{}] Batch[{}]\tSpeed: {:.2f} samples/sec'
                         .format(epoch, store.rank, batch_num, speed))
            logging.info('{} = {:.2f}'.format(*train_metric.get()))

        btic = time.time()
        batch_num += 1

    elapsed = time.time() - tic
    # Print test accuracy after every epoch
    test_accuracy = evaluate(test_data, net, ctx)
    if store.rank == 0:
        logging.info(f"Epoch {epoch}: Test_acc {test_accuracy}")