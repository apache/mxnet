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
"""The module to run training on the Urban sounds dataset"""
from __future__ import print_function
import sys
import os
import time
import warnings
import mxnet as mx
from mxnet import gluon, nd, autograd
from datasets import AudioFolderDataset
import model
sys.path.append('../')

def evaluate_accuracy(data_iterator, net):
    """Function to evaluate accuracy of any data iterator passed to it as an argument"""
    acc = mx.metric.Accuracy()
    for data, label in data_iterator:
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        predictions = predictions.reshape((-1, 1))
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


def train(train_dir=None, train_csv=None, epochs=30, batch_size=32):
    """Function responsible for running the training the model."""

    if not train_dir or not os.path.exists(train_dir) or not train_csv:
        warnings.warn("No train directory could be found ")
        return
    # Make a dataset from the local folder containing Audio data
    print("\nMaking an Audio Dataset...\n")
    tick = time.time()
    aud_dataset = AudioFolderDataset(train_dir, train_csv=train_csv, file_format='.wav', skip_header=True)
    tock = time.time()

    print("Loading the dataset took ", (tock-tick), " seconds.")
    print("\n=======================================\n")
    print("Number of output classes = ", len(aud_dataset.synsets))
    print("\nThe labels are : \n")
    print(aud_dataset.synsets)
    # Get the model to train
    net = model.get_net(len(aud_dataset.synsets))
    print("\nNeural Network = \n")
    print(net)
    print("\nModel - Neural Network Generated!\n")
    print("=======================================\n")

    #Define the loss - Softmax CE Loss
    softmax_loss = gluon.loss.SoftmaxCELoss(from_logits=False, sparse_label=True)
    print("Loss function initialized!\n")
    print("=======================================\n")

    #Define the trainer with the optimizer
    trainer = gluon.Trainer(net.collect_params(), 'adadelta')
    print("Optimizer - Trainer function initialized!\n")
    print("=======================================\n")
    print("Loading the dataset to the Gluon's OOTB Dataloader...")

    #Getting the data loader out of the AudioDataset and passing the transform
    from transforms import MFCC
    aud_transform = MFCC()
    tick = time.time()

    audio_train_loader = gluon.data.DataLoader(aud_dataset.transform_first(aud_transform), batch_size=32, shuffle=True)
    tock = time.time()
    print("Time taken to load data and apply transform here is ", (tock-tick), " seconds.")
    print("=======================================\n")


    print("Starting the training....\n")
    # Training loop
    tick = time.time()
    batch_size = batch_size
    num_examples = len(aud_dataset)

    for epoch in range(epochs):
        cumulative_loss = 0
        for data, label in audio_train_loader:
            with autograd.record():
                output = net(data)
                loss = softmax_loss(output, label)
            loss.backward()

            trainer.step(batch_size)
            cumulative_loss += mx.nd.sum(loss).asscalar()

        if epoch%5 == 0:
            train_accuracy = evaluate_accuracy(audio_train_loader, net)
            print("Epoch {}. Loss: {} Train accuracy : {} ".format(epoch, cumulative_loss/num_examples, train_accuracy))
            print("\n------------------------------\n")

    train_accuracy = evaluate_accuracy(audio_train_loader, net)
    tock = time.time()
    print("\nFinal training accuracy: ", train_accuracy)

    print("Training the sound classification for ", epochs, " epochs, MLP model took ", (tock-tick), " seconds")
    print("====================== END ======================\n")

    print("Trying to save the model parameters here...")
    net.save_parameters("./net.params")
    print("Saved the model parameters in current directory.")


if __name__ == '__main__':
    training_dir = './Train'
    training_csv = './train.csv'
    epochs = 30
    batch_size = 32

    try:
        import argparse
        parser = argparse.ArgumentParser(description="Urban Sounds classification example - MXNet Gluon")
        parser.add_argument('--train', '-t', help="Enter the folder path that contains your audio files", type=str)
        parser.add_argument('--csv', '-c', help="Enter the filename of the csv that contains filename\
        to label mapping", type=str)
        parser.add_argument('--epochs', '-e', help="Enter the number of epochs \
        you would want to run the training for.", type=int)
        parser.add_argument('--batch_size', '-b', help="Enter the batch_size of data", type=int)
        args = parser.parse_args()

        if args:
            if args.train:
                training_dir = args.train

            if args.csv:
                training_csv = args.csv

            if args.epochs:
                epochs = args.epochs

            if args.batch_size:
                batch_size = args.batch_size


    except ImportError as er:
        warnings.warn("Argument parsing module could not be imported \
        Passing default arguments.")


    train(train_dir=training_dir, train_csv=training_csv, epochs=epochs, batch_size=batch_size)
    print("Urban sounds classification Training DONE!")
