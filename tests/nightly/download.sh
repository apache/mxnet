#!/bin/bash

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


dmlc_download() {
    url=http://data.mxnet.io/mxnet/datasets/
    dir=$1
    file=$2
    if [ ! -e data/${dir}/$file ]; then
        wget ${url}/${dir}/${file} -P data/${dir}/ || exit -1
    else
        echo "data/${dir}/$file already exits"
    fi
}

dmlc_download mnist t10k-images-idx3-ubyte
dmlc_download mnist t10k-labels-idx1-ubyte
dmlc_download mnist train-images-idx3-ubyte
dmlc_download mnist train-labels-idx1-ubyte

dmlc_download cifar10 train.rec
dmlc_download cifar10 test.rec
