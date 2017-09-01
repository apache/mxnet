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

# pylint: skip-file
import os, gzip
import pickle as pickle
import sys

# download mnist.pkl.gz
def GetMNIST_pkl():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if not os.path.exists('data/mnist.pkl.gz'):
        os.system("wget -q http://deeplearning.net/data/mnist/mnist.pkl.gz -P data/")

# download ubyte version of mnist and untar
def GetMNIST_ubyte():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if (not os.path.exists('data/train-images-idx3-ubyte')) or \
       (not os.path.exists('data/train-labels-idx1-ubyte')) or \
       (not os.path.exists('data/t10k-images-idx3-ubyte')) or \
       (not os.path.exists('data/t10k-labels-idx1-ubyte')):
        os.system("wget -q http://data.mxnet.io/mxnet/data/mnist.zip -P data/")
        os.chdir("./data")
        os.system("unzip -u mnist.zip")
        os.chdir("..")

# download cifar
def GetCifar10():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if (not os.path.exists('data/cifar/train.rec')) or \
       (not os.path.exists('data/cifar/test.rec')) or \
       (not os.path.exists('data/cifar/train.lst')) or \
       (not os.path.exists('data/cifar/test.lst')):
        os.system("wget -q http://data.mxnet.io/mxnet/data/cifar10.zip -P data/")
        os.chdir("./data")
        os.system("unzip -u cifar10.zip")
        os.chdir("..")
