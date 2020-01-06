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

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --param-file=./model/imagenet1k-resnet-152-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec


python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --param-file=./model/imagenet1k-inception-bn-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
