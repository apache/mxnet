#!/usr/bin/env bash
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
set -euo pipefail

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     CMD='wget -q';;
    Darwin*)    CMD='curl -sO';;
    CYGWIN*)    CMD='wget -q';;
    MINGW*)     CMD='wget -q';;
    *)          CMD=""
esac

mkdir -p data
if [ ! -d "./data/mnist_data" ]; then
    mkdir ./data/mnist_data
    pushd .
    cd data/mnist_data
    echo "Downloading MNIST dataset"
    echo "train-images-idx3-ubyte.gz"
    $CMD https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-images-idx3-ubyte.gz
    echo "train-labels-idx1-ubyte.gz"
    $CMD https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-labels-idx1-ubyte.gz
    echo "t10k-images-idx3-ubyte.gz"
    $CMD https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-images-idx3-ubyte.gz
    echo "t10k-labels-idx1-ubyte.gz"
    $CMD https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-labels-idx1-ubyte.gz
    echo "mnist_train.csv.gz"
    $CMD http://data.mxnet.io/data/mnist_train.csv.gz
    gzip -d *.gz
    popd
fi



