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

# Executes mxnet python images pipeline functions: build, test, publish
# Assumes script is run from the root of the mxnet repository
# Assumes script is being run within MXNet CD infrastructure

set -xe

usage="Usage: python_images.sh <build|test|push> MXNET-VARIANT"

command=${1:?$usage}
mxnet_variant=${2:?$usage}

cd_utils='cd/utils'
ci_utils='ci/'

docker_tags=($(./${cd_utils}/docker_tag.sh ${mxnet_variant}))
main_tag="${docker_tags[0]}_py3"
base_image=$(./${cd_utils}/mxnet_base_image.sh ${mxnet_variant})
repository="python"
image_name="${repository}:${main_tag}"

resources_path='cd/python/docker'

if [ ! -z "${RELEASE_PUBLIC_ECR_REPOSITORY}" ]; then
    image_name="${RELEASE_PUBLIC_ECR_REPOSITORY}/${image_name}"
fi

build() {
    # NOTE: Ensure the correct context root is passed in when building - Dockerfile expects ./wheel_build
    docker build -t "${image_name}" --build-arg BASE_IMAGE="${base_image}" --build-arg MXNET_COMMIT_ID=${GIT_COMMIT} -f ${resources_path}/Dockerfile ./wheel_build
}

test() {
    local runtime_param=""
    if [[ ${mxnet_variant} == cu* ]]; then
        runtime_param="--runtime=nvidia"
    fi
    local test_image_name="${image_name}_test"

    # Ensure the correct context root is passed in when building - Dockerfile.test expects ci directory
    docker build -t "${test_image_name}" --build-arg USER_ID=`id -u` --build-arg GROUP_ID=`id -g` --build-arg BASE_IMAGE="${image_name}" -f ${resources_path}/Dockerfile.test ./ci
}

push() {
    if [ -z "${RELEASE_PUBLIC_ECR_REPOSITORY}" ]; then
        echo "Cannot publish image without RELEASE_PUBLIC_ECR_REPOSITORY environment variable being set."
        exit 1
    fi

    # Retrieve an authentication token and authenticate Docker client to registry
    aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/w6z5f7h2

    # Push image
    docker push "${image_name}"
}

case ${command} in
    "build")
        build
        ;;

    "test")
        test
        ;;

    "push")
        push
        ;;

    *)
        echo $usage
        exit 1
esac
