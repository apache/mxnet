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
""" Prediction module for Urban Sounds Classification"""
from __future__ import print_function
import os
import sys
import warnings
import mxnet as mx
from mxnet import nd
from model import get_net
try:
    import librosa
except ImportError:
    raise ImportError("Librosa is not installed! please run the following command:\
     `pip install librosa`")
sys.path.append('../')

def predict(prediction_dir='./Test'):
    """The function is used to run predictions on the audio files in the directory `pred_directory`.

    Parameters
    ----------
    net:
        The model that has been trained.
    prediction_dir: string, default ./Test
        The directory that contains the audio files on which predictions are to be made

    """

    if not os.path.exists(prediction_dir):
        warnings.warn("The directory on which predictions are to be made is not found!")
        return

    if len(os.listdir(prediction_dir)) == 0:
        warnings.warn("The directory on which predictions are to be made is empty! Exiting...")
        return

    # Loading synsets
    if not os.path.exists('./synset.txt'):
        warnings.warn("The synset or labels for the dataset do not exist. Please run the training script first.")
        return

    with open("./synset.txt", "r") as f:
        synset = [l.rstrip() for l in f]
    net = get_net(len(synset))
    print("Trying to load the model with the saved parameters...")
    if not os.path.exists("./net.params"):
        warnings.warn("The model does not have any saved parameters... Cannot proceed! Train the model first")
        return

    net.load_parameters("./net.params")
    file_names = os.listdir(prediction_dir)
    full_file_names = [os.path.join(prediction_dir, item) for item in file_names]
    from transforms import MFCC
    mfcc = MFCC()
    print("\nStarting predictions for audio files in ", prediction_dir, " ....\n")
    for filename in full_file_names:
        # Argument kaiser_fast to res_type is faster than 'kaiser_best'. To reduce the load time, passing kaiser_fast.
        X1, _ = librosa.load(filename, res_type='kaiser_fast')
        transformed_test_data = mfcc(mx.nd.array(X1))
        output = net(transformed_test_data.reshape((1, -1)))
        prediction = nd.argmax(output, axis=1)
        print(filename, " -> ", synset[(int)(prediction.asscalar())])


if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Urban Sounds clsssification example - MXNet")
        parser.add_argument('--pred', '-p', help="Enter the folder path that contains your audio files", type=str)
        args = parser.parse_args()
        pred_dir = args.pred

    except ImportError:
        warnings.warn("Argparse module not installed! passing default arguments.")
        pred_dir = './Test'
    predict(prediction_dir=pred_dir)
    print("Urban sounds classification Prediction DONE!")
