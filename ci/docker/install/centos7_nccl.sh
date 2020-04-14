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

set -ex


if [ -z ${SHORT_CUDA_VERSION} ]; then
    echo "Error: SHORT_CUDA_VERSION environment variable undefined"
    exit 1
fi
if [ -z ${SHORT_NCCL_VERSION} ]; then
    echo "Error: SHORT_NCCL_VERSION environment variable undefined"
    exit 1
fi

curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm -O
rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
yum check-update || true  # exit code 100 in case of available updates
yum install -y libnccl-${SHORT_NCCL_VERSION}-1+cuda${SHORT_CUDA_VERSION} libnccl-devel-${SHORT_NCCL_VERSION}-1+cuda${SHORT_CUDA_VERSION} libnccl-static-${SHORT_NCCL_VERSION}-1+cuda${SHORT_CUDA_VERSION}
