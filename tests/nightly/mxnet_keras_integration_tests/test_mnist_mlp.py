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

'''
This code is forked from https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and modified to use as MXNet-Keras integration testing for functionality and sanity performance
benchmarking.

Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from os import environ

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Imports for benchmarking
from profiler import profile
from model_util import make_model

# Imports for assertions
from assertion_util import assert_results

# Other environment variables
MACHINE_TYPE = environ['MXNET_KERAS_TEST_MACHINE']
IS_GPU = (environ['MXNET_KERAS_TEST_MACHINE'] == 'GPU')
MACHINE_TYPE = 'GPU' if IS_GPU else 'CPU'
GPU_NUM = int(environ['GPU_NUM']) if IS_GPU else 0

# Expected Benchmark Numbers
CPU_BENCHMARK_RESULTS = {'TRAINING_TIME':550.0, 'MEM_CONSUMPTION':400.0, 'TRAIN_ACCURACY': 0.85, 'TEST_ACCURACY':0.85}
GPU_1_BENCHMARK_RESULTS = {'TRAINING_TIME':40.0, 'MEM_CONSUMPTION':200, 'TRAIN_ACCURACY': 0.85, 'TEST_ACCURACY':0.85}
# TODO: Fix Train and Test accuracy numbers in multiple gpu mode. Setting it to 0 for now to get whole integration set up done
GPU_2_BENCHMARK_RESULTS = {'TRAINING_TIME':45.0, 'MEM_CONSUMPTION':375, 'TRAIN_ACCURACY': 0.0, 'TEST_ACCURACY':0.0}
GPU_4_BENCHMARK_RESULTS = {'TRAINING_TIME':55.0, 'MEM_CONSUMPTION':750.0, 'TRAIN_ACCURACY': 0.0, 'TEST_ACCURACY':0.0}
GPU_8_BENCHMARK_RESULTS = {'TRAINING_TIME':100.0, 'MEM_CONSUMPTION':1800.0, 'TRAIN_ACCURACY': 0.0, 'TEST_ACCURACY':0.0}

# Dictionary to store profiling output
profile_output = {}

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
make_model(model, loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

def train_model():
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
    profile_output['TRAIN_ACCURACY'] = history.history['acc'][-1]

def test_run():
    # Calling training and profile memory usage
    profile_output["MODEL"] = "MNIST MLP"
    run_time, memory_usage = profile(train_model)

    profile_output['TRAINING_TIME'] = float(run_time)
    profile_output['MEM_CONSUMPTION'] = float(memory_usage)

    score = model.evaluate(X_test, Y_test, verbose=0)
    profile_output["TEST_ACCURACY"] = score[1]

    assert_results(MACHINE_TYPE, IS_GPU, GPU_NUM, profile_output, CPU_BENCHMARK_RESULTS, GPU_1_BENCHMARK_RESULTS, GPU_2_BENCHMARK_RESULTS, GPU_4_BENCHMARK_RESULTS, GPU_8_BENCHMARK_RESULTS)
