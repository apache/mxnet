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

# Check Params
programname=$0

function usage {
    echo "usage: $programname [version] [pip_tag] [path]"
    echo "  [version]  Mxnet Version to build"
    echo "  [pip_tag]  Pip Tag to use"
    echo "  [path]     Path to MXNet repository (to run tests)"
    echo " "
    exit 1
}

if [ $# -le 2 ] || [ $# -ge 4 ]
then
    usage
    exit 1
fi

# Two params provided
echo "Building Docker Images for Apache MXNet (Incubating) v$1"
mxnet_version="${1}"
pip_tag="${2}"
test_dir="${3}"

# Remove the logs directory if it already exists else it may fail due to old logs.
LOGDIR=~/temp/docker_logs
if [ -d "${LOGDIR}" ]; then
  rm -rf ${LOGDIR}
fi

# Create ~/temp if it does not exist
mkdir -p ~/temp
mkdir ${LOGDIR}


# Functions
docker_test_image_cpu(){
    image_tag="${1}"
    python_version="${2}"
    echo "Running tests on mxnet/python:${image_tag}"
    docker run -v ${test_dir}:/mxnet mxnet/python:${image_tag} bash -c "${python_version} /mxnet/docker/docker-python/test_mxnet.py ${mxnet_version}"
    docker run -v ${test_dir}:/mxnet mxnet/python:${image_tag} bash -c "${python_version} /mxnet/tests/python/train/test_conv.py"
    docker run -v ${test_dir}:/mxnet mxnet/python:${image_tag} bash -c "${python_version} /mxnet/example/image-classification/train_mnist.py"
}

docker_test_image_gpu(){
    image_tag="${1}"
    python_version="${2}"
    echo "Running tests on mxnet/python:${1}"
    nvidia-docker run -v ${test_dir}:/mxnet mxnet/python:${image_tag} bash -c "${python_version} /mxnet/docker/docker-python/test_mxnet.py ${mxnet_version}"
    nvidia-docker run -v ${test_dir}:/mxnet mxnet/python:${image_tag} bash -c "${python_version} /mxnet/tests/python/train/test_conv.py --gpu"
    nvidia-docker run -v ${test_dir}:/mxnet mxnet/python:${image_tag} bash -c "${python_version} /mxnet/example/image-classification/train_mnist.py --gpus 0,1,2,3"
}

# if both $MXNET_DOCKERHUB_PASSWORD and $MXNET_DOCKERHUB_USERNAME environment variables are set, docker will automatically login
# if env variables are not set, login will be interactive.
docker_account_login(){
    if [[ -z $MXNET_DOCKERHUB_PASSWORD ]] || [[ -z $MXNET_DOCKERHUB_USERNAME ]]; then
        docker login
    else
        echo $MXNET_DOCKERHUB_PASSWORD | docker login -u $MXNET_DOCKERHUB_USERNAME --password-stdin
    fi
}

docker_account_logout(){
    docker logout
}

docker_push_image(){
    image_tag="${1}"
    docker push mxnet/python:${image_tag}
}

docker_generate_image_cpu(){
    image_tag="${1}"
    dockerfile="${2}"
    python_version="${3}"
    echo "Building docker image mxnet/python:${image_tag}"
    docker build --build-arg version=${pip_tag} -t mxnet/python:${image_tag} -f ${dockerfile} .
    docker_test_image_cpu ${image_tag} ${python_version}
}

docker_tag_image_cpu(){
    original_tag="${1}"
    image_tag="${2}"
    python_version="${3}"
    docker tag mxnet/python:${original_tag} mxnet/python:${image_tag}
    docker_test_image_cpu ${image_tag} ${python_version}
}

docker_generate_image_gpu(){
    image_tag="${1}"
    dockerfile="${2}"
    python_version="${3}"
    echo "Building docker image mxnet/python:${1}"
    docker build --build-arg version=${pip_tag} -t mxnet/python:${image_tag} -f ${dockerfile} .
    docker_test_image_gpu ${image_tag} ${python_version}
}

docker_tag_image_gpu(){
    original_tag="${1}"
    image_tag="${2}"
    python_version="${3}"
    docker tag mxnet/python:${original_tag} mxnet/python:${image_tag}
    docker_test_image_gpu ${image_tag} ${python_version}
}

check_errors(){
    egrep -i "not found|error|returned a non-zero code|fail" ${LOGDIR}/docker*
    if [ $? -eq 0 ]; then
        echo "ERROR: One of the build/test commands failed. Refer to the filename above to see which image tag caused it."
        exit 1
    else
        echo "Success: No errors found"
    fi
}

# Build and Test dockerfiles - CPU
docker_generate_image_cpu "${mxnet_version}_cpu" "Dockerfile.mxnet.python.cpu" "python"  > ${LOGDIR}/docker_cpu.out 2>&1 &
docker_generate_image_cpu "${mxnet_version}_cpu_mkl" "Dockerfile.mxnet.python.cpu.mkl" "python" > ${LOGDIR}/docker_cpu_mkl.out 2>&1 &


#Build and Test dockerfiles - GPU
docker_generate_image_gpu "${mxnet_version}_gpu_cu90" "Dockerfile.mxnet.python.gpu.cu90" "python" > ${LOGDIR}/docker_gpu_cu90.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu90_mkl" "Dockerfile.mxnet.python.gpu.cu90.mkl" "python" > ${LOGDIR}/docker_gpu_cu90_mkl.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu80" "Dockerfile.mxnet.python.gpu.cu80" "python" > ${LOGDIR}/docker_gpu_cu80.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu80_mkl" "Dockerfile.mxnet.python.gpu.cu80.mkl" "python" > ${LOGDIR}/docker_gpu_cu80_mkl.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu92" "Dockerfile.mxnet.python.gpu.cu92" "python" > ${LOGDIR}/docker_gpu_cu92.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu92_mkl" "Dockerfile.mxnet.python.gpu.cu92.mkl" "python" > ${LOGDIR}/docker_gpu_cu92_mkl.out 2>&1

echo "Waiting for MXNet Python2 Docker Images to Build"
wait

# Build and Test Python3 dockerfiles - CPU
docker_generate_image_cpu "${mxnet_version}_cpu_py3" "Dockerfile.mxnet.python3.cpu" "python3" > ${LOGDIR}/docker_cpu_py3.out 2>&1 &
docker_generate_image_cpu "${mxnet_version}_cpu_mkl_py3" "Dockerfile.mxnet.python3.cpu.mkl" "python3" > ${LOGDIR}/docker_cpu_mkl_py3.out 2>&1 &

#Build and Test Python3 dockerfiles - GPU
docker_generate_image_gpu "${mxnet_version}_gpu_cu90_py3" "Dockerfile.mxnet.python3.gpu.cu90" "python3" > ${LOGDIR}/docker_gpu_cu90_py3.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu90_mkl_py3" "Dockerfile.mxnet.python3.gpu.cu90.mkl" "python3" > ${LOGDIR}/docker_gpu_cu90_mkl_py3.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu80_py3" "Dockerfile.mxnet.python3.gpu.cu80" "python3" > ${LOGDIR}/docker_gpu_cu80_py3.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu80_mkl_py3" "Dockerfile.mxnet.python3.gpu.cu80.mkl" "python3" > ${LOGDIR}/docker_gpu_cu80_mkl_py3.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu92_py3" "Dockerfile.mxnet.python3.gpu.cu92" "python3" > ${LOGDIR}/docker_gpu_cu92_py3.out 2>&1 &
docker_generate_image_gpu "${mxnet_version}_gpu_cu92_mkl_py3" "Dockerfile.mxnet.python3.gpu.cu92.mkl" "python3" > ${LOGDIR}/docker_gpu_cu92_mkl_py3.out 2>&1

echo "Waiting for MXNet Python3 Docker Images to Build"
wait

echo "Re-Tag 6 images with version-free names (for Benchmarking) - only after previous builds complete. "
docker_tag_image_cpu "${mxnet_version}_cpu" "latest" "python" > ${LOGDIR}/docker_latest.out 2>&1 &
docker_tag_image_gpu "${mxnet_version}_gpu_cu90" "gpu" "python" > ${LOGDIR}/docker_gpu.out 2>&1 &
docker_tag_image_cpu "${mxnet_version}_cpu_mkl" "latest_cpu_mkl_py2" "python" > ${LOGDIR}/docker_latest_cpu_mkl_py2.out 2>&1 &
docker_tag_image_cpu "${mxnet_version}_cpu_mkl_py3" "latest_cpu_mkl_py3" "python3" > ${LOGDIR}/docker_latest_cpu_mkl_py3.out 2>&1 &
docker_tag_image_gpu "${mxnet_version}_gpu_cu90_mkl" "latest_gpu_mkl_py2" "python" > ${LOGDIR}/docker_latest_gpu_mkl_py2.out 2>&1 &
docker_tag_image_gpu "${mxnet_version}_gpu_cu90_mkl_py3" "latest_gpu_mkl_py3" "python3" > ${LOGDIR}/docker_latest_gpu_mkl_py3.out 2>&1
wait

# Parse all the docker logfiles to make sure there is no error. Fail script if error is found.
check_errors

# Push dockerfiles
echo "All images were successfully built. Now login to dockerhub and push images"
docker_account_login

# Python2
docker_push_image "${mxnet_version}_cpu"
docker_push_image "${mxnet_version}_cpu_mkl"
docker_push_image "${mxnet_version}_gpu_cu90"
docker_push_image "${mxnet_version}_gpu_cu90_mkl"
docker_push_image "${mxnet_version}_gpu_cu80"
docker_push_image "${mxnet_version}_gpu_cu80_mkl"
docker_push_image "${mxnet_version}_gpu_cu92"
docker_push_image "${mxnet_version}_gpu_cu92_mkl"

# Python3
docker_push_image "${mxnet_version}_cpu_py3"
docker_push_image "${mxnet_version}_cpu_mkl_py3"
docker_push_image "${mxnet_version}_gpu_cu90_py3"
docker_push_image "${mxnet_version}_gpu_cu90_mkl_py3"
docker_push_image "${mxnet_version}_gpu_cu80_py3"
docker_push_image "${mxnet_version}_gpu_cu80_mkl_py3"
docker_push_image "${mxnet_version}_gpu_cu92_py3"
docker_push_image "${mxnet_version}_gpu_cu92_mkl_py3"

docker_push_image "latest"
docker_push_image "gpu"
docker_push_image "latest_cpu_mkl_py2"
docker_push_image "latest_cpu_mkl_py3"
docker_push_image "latest_gpu_mkl_py2"
docker_push_image "latest_gpu_mkl_py3"


docker_account_logout

echo "Successfully Built, Tested and Pushed all Images to Dockerhub. Link: https://hub.docker.com/r/mxnet/python/tags/"

#Delete the log directory since everything succeeded:
rm -rf ${LOGDIR}