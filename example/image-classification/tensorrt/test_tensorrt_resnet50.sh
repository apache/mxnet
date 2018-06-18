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

EPOCH=20
MODEL_PREFIX="resnet50"
SYMBOL="${MODEL_PREFIX}-symbol.json"
PARAMS="${MODEL_PREFIX}-$(printf "%04d" $EPOCH).params"
DATA_DIR="./data"

if [[ ! -f $SYMBOL || ! -f $PARAMS ]]; then
  echo -e "\nTrained model does not exist. Training - please wait...\n"
  python $MXNET_HOME/example/image-classification/train_cifar10.py \
     --network resnet --num-layers 50 --num-epochs ${EPOCH} \
     --model-prefix ./${MODEL_PREFIX} --gpus 0
else
   echo "Pre-trained model exists. Skipping training."
fi

echo "Running inference script."

python test_tensorrt_resnet50.py $MODEL_PREFIX $EPOCH $DATA_DIR

