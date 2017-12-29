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


MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
OS=$(uname)
if [ "$OS" = "Darwin" ]; then
  CLASS_PATH=$MXNET_ROOT/scala-package/assembly/osx-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*
else
  CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*
fi
# which gpu card to use, -1 means cpu
GPU=$1
# you can get the training data file using the following command
# wget http://data.mxnet.io/data/char_lstm.zip
# unzip -o char_lstm.zip
# for example ./datas/obama.txt
DATA_PATH=$2
# for example ./models
SAVE_MODEL_PATH=$3

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.rnn.TrainCharRnn \
	--data-path $DATA_PATH \
	--save-model-path $SAVE_MODEL_PATH \
	--gpu $GPU \
