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

CURL_OPTIONS='--connect-timeout 5 --max-time 10 --retry 3 --retry-delay 0 --retry-max-time 40 -L'

mkdir -p data/mnist_data
cd data/mnist_data

curl ${CURL_OPTIONS} https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-images-idx3-ubyte.gz \
    -o train-images-idx3-ubyte.gz

curl ${CURL_OPTIONS} https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-labels-idx1-ubyte.gz \
    -o train-labels-idx1-ubyte.gz

curl ${CURL_OPTIONS} https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-images-idx3-ubyte.gz \
    -o t10k-images-idx3-ubyte.gz

curl ${CURL_OPTIONS} https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-labels-idx1-ubyte.gz \
    -o t10k-labels-idx1-ubyte.gz

curl ${CURL_OPTIONS} http://data.mxnet.io/data/mnist_train.csv.gz \
    -o mnist_train.csv.gz
gzip -d *.gz
