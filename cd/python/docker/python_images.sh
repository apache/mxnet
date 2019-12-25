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

usage="Usage: python_images.sh <build|test|publish> MXNET-VARIANT <py2|py3>"

command=${1:?$usage}
mxnet_variant=${2:?$usage}
python_version=${3:?usage}

cd_utils='cd/utils'
ci_utils='ci/'

case ${python_version} in
    py3)
        python_cmd="python3"
        ;;
    py2)
        python_cmd="python"
        ;;
    *)
        echo "Error: specify python version with either 'py2' or 'py3'"
        exit 1
        ;;
esac

docker_tags=($(./${cd_utils}/docker_tag.sh ${mxnet_variant}))
main_tag="${docker_tags[0]}_${python_version}"
base_image=$(./${cd_utils}/mxnet_base_image.sh ${mxnet_variant})
repository="python"
image_name="${repository}:${main_tag}"

resources_path='cd/python/docker'

if [ ! -z "${RELEASE_DOCKERHUB_REPOSITORY}" ]; then
    image_name="${RELEASE_DOCKERHUB_REPOSITORY}/${image_name}"
fi

build() {
    # NOTE: Ensure the correct context root is passed in when building - Dockerfile expects ./wheel_build
    docker build -t "${image_name}" --build-arg PYTHON_CMD=${python_cmd} --build-arg BASE_IMAGE="${base_image}" --build-arg MXNET_COMMIT_ID=${GIT_COMMIT} -f ${resources_path}/Dockerfile ./wheel_build
}

test() {
    local runtime_param=""
    if [[ ${mxnet_variant} == cu* ]]; then
        runtime_param="--runtime=nvidia"
    fi
    local test_image_name="${image_name}_test"
    
    # Ensure the correct context root is passed in when building - Dockerfile.test expects ci directory
    docker build -t "${test_image_name}" --build-arg USER_ID=`id -u` --build-arg GROUP_ID=`id -g` --build-arg BASE_IMAGE="${image_name}" -f ${resources_path}/Dockerfile.test ./ci
    ./ci/safe_docker_run.py ${runtime_param} --cap-add "SYS_PTRACE" -u `id -u`:`id -g` -v `pwd`:/work/mxnet "${test_image_name}" ${resources_path}/test_python_image.sh "${mxnet_variant}" "${python_cmd}"
}

push() {
    if [ -z "${RELEASE_DOCKERHUB_REPOSITORY}" ]; then
        echo "Cannot publish image without RELEASE_DOCKERHUB_REPOSITORY environment variable being set."
        exit 1
    fi

    # The secret name env var is set in the Jenkins configuration
    # Manage Jenkins -> Configure System
    ./${ci_utils}/docker_login.py --secret-name "${RELEASE_DOCKERHUB_SECRET_NAME}"

    # Push image
    docker push "${image_name}"

    # Iterate over remaining tags, if any
    for ((i=1;i<${#docker_tags[@]};i++)); do
        local docker_tag="${docker_tags[${i}]}"
        local latest_image_name="${RELEASE_DOCKERHUB_REPOSITORY}/${repository}:${docker_tag}"

        # latest and latest gpu should only be pushed for py3
        if [[ ${docker_tag} == "latest" || ${docker_tag} == "latest_gpu" ]]; then
            if [[ ${python_version} == "py2" ]]; then
                continue
            fi
        else
            latest_image_name="${latest_image_name}_${python_version}"
        fi

        docker tag "${image_name}" "${latest_image_name}"
        docker push "${latest_image_name}"
        echo "Successfully pushed ${latest_image_name}. Pull it with:"
        echo "docker pull ${latest_image_name}"
        echo "For a complete list of tags see https://hub.docker.com/u/${RELEASE_DOCKERHUB_REPOSITORY}/${repository}"
    done    
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
