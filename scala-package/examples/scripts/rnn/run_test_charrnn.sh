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

# you can get the training data file using the following command
# wget http://data.mxnet.io/data/char_lstm.zip
# unzip -o char_lstm.zip
# for example ./datas/obama.txt
DATA_PATH=$1
# for example ./models/obama
MODEL_PREFIX=$2
# feel free to change the starter sentence
STARTER_SENTENCE="The joke"

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.rnn.TestCharRnn \
	--data-path $DATA_PATH \
	--model-prefix $MODEL_PREFIX \
	--starter-sentence "$STARTER_SENTENCE"
