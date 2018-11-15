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

"""
    Urban Sounds Dataset:

    To be able to run this example:

    1. Download the dataset(train.zip, test.zip) required for this example from the location:
    **https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU**
    2. Extract both the zip archives into the **current directory** -
       after unzipping you would get 2 new folders namely,\
       **Train** and **Test** and two csv files - **train_csv.csv**, **test_csv.csv**
    3. Apache MXNet is installed on the machine. For instructions, go to the link:
    **https://mxnet.incubator.apache.org/install/ **
    4. Librosa is installed. To install, follow the instructions here:
     **https://librosa.github.io/librosa/install.html**

"""
import os
import time
import warnings
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon.contrib.data.audio.datasets import AudioFolderDataset
from mxnet.gluon.contrib.data.audio.transforms import MFCC
try:
    import argparse
except ImportError as er:
    warnings.warn("Argument parsing module could not be imported and hence \
    no arguments passed to the script can actually be parsed.")
try:
    import librosa
except ImportError as er:
    warnings.warn("ALibrosa module could not be imported and hence \
    audio could not be loaded onto numpy array.")


# Defining a neural network with number of labels
def get_net(num_labels=10):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(256, activation="relu")) # 1st layer (256 nodes)
        net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    net.add(gluon.nn.Dense(num_labels))
    net.collect_params().initialize(mx.init.Normal(1.))
    return net


# Defining a function to evaluate accuracy
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for _, (data, label) in enumerate(data_iterator):
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        predictions = predictions.reshape((-1, 1))
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


def train(train_dir=None, pred_directory='./Test', train_csv=None, epochs=30, batch_size=32):
    """
        The function responsible for running the training the model.
    """
    if not train_dir or not os.path.exists(train_dir) or not train_csv:
        warnings.warn("No train directory could be found ")
        return
    # Make a dataset from the local folder containing Audio data
    print("\nMaking an Audio Dataset...\n")
    tick = time.time()
    aud_dataset = AudioFolderDataset('./Train', has_csv=True, train_csv='./train.csv', file_format='.wav', skip_rows=1)
    tock = time.time()

    print("Loading the dataset took ", (tock-tick), " seconds.")
    print("\n=======================================\n")
    print("Number of output classes = ", len(aud_dataset.synsets))
    print("\nThe labels are : \n")
    print(aud_dataset.synsets)
    # Get the model to train
    net = get_net(len(aud_dataset.synsets))
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
    aud_transform = gluon.data.vision.transforms.Compose([MFCC()])
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

    for e in range(epochs):
        cumulative_loss = 0
        for _, (data, label) in enumerate(audio_train_loader):
            with autograd.record():
                output = net(data)
                loss = softmax_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += mx.nd.sum(loss).asscalar()

        if e%5 == 0:
            train_accuracy = evaluate_accuracy(audio_train_loader, net)
            print("Epoch %s. Loss: %s Train accuracy : %s " % (e, cumulative_loss/num_examples, train_accuracy))
            print("\n------------------------------\n")

    train_accuracy = evaluate_accuracy(audio_train_loader, net)
    tock = time.time()
    print("\nFinal training accuracy: ", train_accuracy)

    print("Training the sound classification for ", epochs, " epochs, MLP model took ", (tock-tick), " seconds")
    print("====================== END ======================\n")
    predict(net, aud_transform, aud_dataset.synsets, pred_directory=pred_directory)


def predict(net, audio_transform, synsets, pred_directory='./Test'):
    """
        The function is used to run predictions on the audio files in the directory `pred_directory`

    Parameters
    ----------
    Keyword arguments that can be passed, which are utilized by librosa module are:
    net: The model that has been trained.

    pred_directory: string, default ./Test
       The directory that contains the audio files on which predictions are to be made
    """
    if not librosa:
        warnings.warn("Librosa dependency not installed! Cnnot load the audio to make predictions. Exitting.")
        return

    if not os.path.exists(pred_directory):
        warnings.warn("The directory on which predictions are to be made is not found!")
        return

    if len(os.listdir(pred_directory)) == 0:
        warnings.warn("The directory on which predictions are to be made is empty! Exitting...")
        return

    file_names = os.listdir(pred_directory)
    full_file_names = [os.path.join(pred_directory, item) for item in file_names]

    print("\nStarting predictions for audio files in ", pred_directory, " ....\n")
    for filename in full_file_names:
        X1, _ = librosa.load(filename, res_type='kaiser_fast')
        transformed_test_data = audio_transform(mx.nd.array(X1))
        output = net(transformed_test_data.reshape((1, -1)))
        prediction = nd.argmax(output, axis=1)
        print(filename, " -> ", synsets[(int)(prediction.asscalar())])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Urban Sounds clsssification example - MXNet")
    parser.add_argument('--train', '-t', help="Enter the folder path that contains your audio files", type=str)
    parser.add_argument('--csv', '-c', help="Enter the filename of the csv that contains filename\
    to label mapping", type=str)
    parser.add_argument('--epochs', '-e', help="Enter the number of epochs \
    you would want to run the training for.", type=int)
    parser.add_argument('--batch_size', '-b', help="Enter the batch_size of data", type=int)
    parser.add_argument('--pred', '-p', help="Enter the folder path that contains your audio \
    files for which you would want to make predictions on.", type=str)
    args = parser.parse_args()
    pred_directory = args.pred

    if args:
        if args.train:
            train_dir = args.train
        else:
            train_dir = './Train'

        if args.csv:
            train_csv = args.csv
        else:
            train_csv = './train.csv'

        if args.epochs:
            epochs = args.epochs
        else:
            epochs = 35

        if args.batch_size:
            batch_size = args.batch_size
        else:
            batch_size = 32
    train(train_dir=train_dir, train_csv=train_csv, epochs=epochs, batch_size=batch_size, pred_directory=pred_directory)
    print("Urban sounds classification DONE!")
