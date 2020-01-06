#!/bin/sh

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

set -ex

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-mode=none

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=naive

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=entropy


python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-mode=none

python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=naive

python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=entropy
