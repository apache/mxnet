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
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# more details please refer to
# https://github.com/Ldpe2G/mxnet/blob/develop/example/neural-style/end_to_end/README.md
TRAIN_DATA_PATH=$1
STYLE_IMG=$2
VGG_MODEL_PATH=$3
SAVE_MODEL_DIR=$4
GPU=0

java -Xmx1024m -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.neuralstyle.end2end.BoostTrain \
	--data-path $TRAIN_DATA_PATH  \
	--vgg--model-path  $VGG_MODEL_PATH \
	--save--model-path $SAVE_MODEL_DIR \
	--style-image $STYLE_IMG \
	--gpu $GPU
