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
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     CMD='wget';;
    Darwin*)    CMD='curl -o';;
    CYGWIN*)    CMD='wget';;
    MINGW*)     CMD='wget';;
    *)          CMD=""
esac

if [ ! -d "./data" ]; then
    mkdir data
fi

if [ ! -d "./data/mnist_data" ]; then
  mkdir ./data/mnist_data

  (cd data/mnist_data; $CMD train-images-idx3-ubyte.gz https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-images-idx3-ubyte.gz)
  (cd data/mnist_data; $CMD train-labels-idx1-ubyte.gz https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-labels-idx1-ubyte.gz)
  (cd data/mnist_data; $CMD t10k-images-idx3-ubyte.gz  https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-images-idx3-ubyte.gz)
  (cd data/mnist_data; $CMD t10k-labels-idx1-ubyte.gz  https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-labels-idx1-ubyte.gz)
  (cd data/mnist_data; $CMD mnist_train.csv.gz         http://data.mxnet.io/data/mnist_train.csv.gz)
  (cd data/mnist_data; gzip -d *.gz)
fi



