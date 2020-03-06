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

# To be run _within_ a runtime image
# Tests the Runtime docker image
# Assumes the mxnet source directory is mounted on /mxnet and cwd is /mxnet

set -ex

# Variant parameter should be passed in
mxnet_variant=${1:?"Missing mxnet variant"}

if [ -z "${MXNET_COMMIT_ID}" ]; then
    echo "MXNET_COMMIT_ID environment variable is empty. Please rebuild the image with MXNET_COMMIT_ID build-arg specified."
    exit 1
fi

# Execute tests
if [[ $mxnet_variant == cu* ]]; then
    mnist_params="--gpu 0"
    test_conv_params="--gpu"
fi

if [[ $mxnet_variant == cpu ]]; then
    python3 tests/python/mkl/test_mkldnn.py
fi

python3 tests/python/train/test_conv.py ${test_conv_params}
python3 example/image-classification/train_mnist.py ${mnist_params}

