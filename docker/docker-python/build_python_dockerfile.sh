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

# Script to take two parameters
echo "Building Docker Images for Apache MXNet (Incubating) v$1"
test_dir="/Users/mbaijal/Documents/mxnet/mxnet_repo2/incubator-mxnet"
#test_dir="${2}"

# function to exit script if command fails
runme() {
        cmd=$*
        echo "$cmd"
        $cmd
        ret=$?
        if [[ ${ret} != 0 ]]; then
                echo " "
                echo "ERROR: Return value non-zero for: $cmd"
                echo " "
                exit 1
        fi
}

docker_build_image(){
    runme docker build -t mxnet/python:${1} -f ${2} .
}

docker_tag_image(){
    docker tag mxnet/python:${1} ${2}
}

docker_test_image_cpu(){
    docker run -v ${test_dir}:/mxnet mxnet/python:${1} bash -c "python /mxnet/docker/docker-python/test_mxnet.py"
    docker run -v ${test_dir}:/mxnet mxnet/python:${1} bash -c "python /mxnet/tests/python/train/test_conv.py"
    docker run -v ${test_dir}:/mxnet mxnet/python:${1} bash -c "python /mxnet/example/image-classification/train_mnist.py"
}

docker_test_image_gpu(){
    nvidia-docker run -v ${test_dir}:/mxnet mxnet/python:${1} bash -c "python /mxnet/docker/docker-python/test_mxnet.py'"
    nvidia-docker run -v ${test_dir}:/mxnet mxnet/python:${1} bash -c "python /mxnet/tests/python/train/test_conv.py --gpu"
    nvidia-docker run -v ${test_dir}:/mxnet mxnet/python:${1} bash -c "python /mxnet/example/image-classification/train_mnist.py"
}

docker_account_login(){
    docker login
}

docker_account_logout(){
    docker logout
}

docker_push_image(){
    runme docker push mxnet/python:${1}
}


# Build and Test dockerfiles - CPU
docker_build_image "${1}_cpu" "Dockerfile.mxnet.python.cpu"
docker_test_image_cpu "${1}_cpu"

docker_build_image "${1}_cpu_mkl" "Dockerfile.mxnet.python.cpu.mkl"
docker_test_image_cpu "${1}_cpu_mkl"

docker_tag_image "${1}_cpu" "latest"
docker_test_image_cpu "latest"


#Build and Test dockerfiles - GPU
docker_build_image "${1}_gpu_cu90" "Dockerfile.mxnet.python.gpu"
docker_test_image_gpu "${1}_gpu_cu90"

docker_build_image "${1}_gpu_mkl_cu90" "Dockerfile.mxnet.python.gpu.mkl"
docker_test_image_gpu "${1}_gpu_cu90"

docker_tag_image "${1}_gpu" "gpu"
docker_test_image_gpu "${1}_gpu_cu90"


# Push dockerfiles
docker_account_login

docker_push_image "${1}_cpu"
docker_push_image "${1}_cpu_mkl"
docker_push_image "latest"
docker_push_image "${1}_gpu_cu90"
docker_push_image "${1}_gpu_cu90"
docker_push_image "${1}_gpu_cu90"

docker_account_logout