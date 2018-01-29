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

#
# Execute command within a docker container
#
# Usage: ci_build.sh <CONTAINER_TYPE> [--dockerfile <DOCKERFILE_PATH>] [-it]
#                    <COMMAND>
#
# CONTAINER_TYPE: Type of the docker container used the run the build: e.g.,
#                 (cpu | gpu)
#
# DOCKERFILE_PATH: (Optional) Path to the Dockerfile used for docker build.  If
#                  this optional value is not supplied (via the --dockerfile
#                  flag), will use Dockerfile.CONTAINER_TYPE in default
#
# COMMAND: Command to be executed in the docker container
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the command line arguments.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Dockerfile to be used in docker build
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}"
DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"

if [[ "$1" == "--dockerfile" ]]; then
    DOCKERFILE_PATH="$2"
    DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
    echo "Using custom Dockerfile path: ${DOCKERFILE_PATH}"
    echo "Using custom docker build context path: ${DOCKER_CONTEXT_PATH}"
    shift 2
fi

if [[ "$1" == "-it" ]]; then
    CI_DOCKER_EXTRA_PARAMS+=('-it')
    shift 1
fi

if [[ "$1" == "--dockerbinary" ]]; then
    DOCKER_BINARY="$2"
    echo "Using custom Docker Engine: ${DOCKER_BINARY}"
    shift 2
fi

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
    exit 1
fi

COMMAND=("$@")

# Validate command line arguments.
if [ "$#" -lt 1 ] || [ ! -e "${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}" ]; then
    supported_container_types=$( ls -1 ${SCRIPT_DIR}/Dockerfile.* | \
        sed -n 's/.*Dockerfile\.\([^\/]*\)/\1/p' | tr '\n' ' ' )
      echo "Usage: $(basename $0) CONTAINER_TYPE COMMAND"
      echo "       CONTAINER_TYPE can be one of [${supported_container_types}]"
      echo "       COMMAND is a command (with arguments) to run inside"
      echo "               the container."
      exit 1
fi

# Only set docker binary automatically if it has not been specified
if [[ -z "${DOCKER_BINARY}" ]]; then
    # Use nvidia-docker if the container is GPU.
    if [[ "${CONTAINER_TYPE}" == *"gpu"* ]]; then
        DOCKER_BINARY="nvidia-docker"
    else
        DOCKER_BINARY="docker"
    fi
    echo "Automatically assuming ${DOCKER_BINARY} as docker binary"
fi

# Helper function to traverse directories up until given file is found.
function upsearch () {
    test / == "$PWD" && return || \
        test -e "$1" && echo "$PWD" && return || \
        cd .. && upsearch "$1"
}

# Set up WORKSPACE. Jenkins will set them for you or we pick
# reasonable defaults if you run it outside of Jenkins.
WORKSPACE="${WORKSPACE:-${SCRIPT_DIR}/../../}"

# Set up a default WORKDIR, unless otherwise specified.
if [[ -z "${WORKDIR}" ]]; then
    WORKDIR="/workspace"
fi

# Determine the docker image name
DOCKER_IMG_NAME="mx-ci.${CONTAINER_TYPE}"

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Skip with_the_same_user for non-linux
uname=`uname`
if [[ "$uname" == "Linux" ]]; then
    # By convention the root of our source dir is always mapped to the /workspace folder
    # inside the docker container.  Our working directory when we start the container
    # is variable, so we should ensure we call commands such as with_the_same_user
    # with their absolute paths.
    PRE_COMMAND="/workspace/tests/ci_build/with_the_same_user"
else
    PRE_COMMAND=""
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "WORKDIR: ${WORKDIR}"
echo "CI_DOCKER_EXTRA_PARAMS: ${CI_DOCKER_EXTRA_PARAMS[@]}"
echo "COMMAND: ${COMMAND[@]}"
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "NODE_NAME: ${NODE_NAME}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMG_NAME}"
echo "PRE_COMMAND: ${PRE_COMMAND}"
echo ""


# Build the docker container.
echo "Building container (${DOCKER_IMG_NAME})..."
docker build -t ${DOCKER_IMG_NAME} \
    -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

# Check docker build status
if [[ $? != "0" ]]; then
    echo "ERROR: docker build failed."
    exit 1
fi


# Run the command inside the container.
echo "Running '${COMMAND[@]}' inside ${DOCKER_IMG_NAME}..."

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).

# Turning off MXNET_STORAGE_FALLBACK_LOG_WARNING temporarily per this issue:
# https://github.com/apache/incubator-mxnet/issues/8980

# By convention always map the root of the MXNet source directory into /workspace.
# ${WORKSPACE} represents the path to the source folder on the host system.
# ${WORKDIR} is the working directory we start the container in.  By default this
# is /workspace, but for example when running cmake it is sometimes /workspace/build.

${DOCKER_BINARY} run --rm --pid=host \
    -v ${WORKSPACE}:/workspace \
    -w ${WORKDIR} \
    -e "CI_BUILD_HOME=${WORKSPACE}" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "CUDA_ARCH=-gencode arch=compute_52,code=[sm_52,compute_52] --fatbin-options -compress-all" \
    -e "MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0" \
    ${CI_DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMG_NAME} \
    ${PRE_COMMAND} \
    ${COMMAND[@]}
