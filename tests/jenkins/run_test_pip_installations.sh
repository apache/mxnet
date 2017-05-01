#!/bin/bash

set -e

if (( $# < 1 )); then
    echo ""
    echo "Usage: $(basename $0) WORKSPACE_PATH"
    echo ""
    exit 1
fi

WORKSPACE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )

PYTHON_VERSIONS=('2.7' '3.4' '3.6' '3.5')
DEVICES=('pip_cu75' 'pip_cu80' 'pip_cpu')

CI_BUILD_DIR=tests/ci_build/pip_tests
# build Docker images and test pip installation for each device
for DEV in "${DEVICES[@]}"; do

    # get Docker binary
    if [[ "${DEV}" == *"cpu"* ]]; then
        DOCKER_BINARY="docker"
    else
        DOCKER_BINARY="nvidia-docker"
    fi

    # concatenate the Dockerfile with dependencies into the device file
    DOCKERFILE="Dockerfile.${DEV}"
    DOCKERFILE_DEVICE="Dockerfile.in.${DEV}"
    rm -rf ${DOCKERFILE}
    cp ${CI_BUILD_DIR}/${DOCKERFILE_DEVICE} ${DOCKERFILE}
    cat ${CI_BUILD_DIR}/Dockerfile.pip_dependencies >> ${DOCKERFILE}

    # build Docker image
    DOCKER_TAG="mxnet/${DEV}"
    ${DOCKER_BINARY} build -t ${DOCKER_TAG} -f ${DOCKERFILE} .

    # test each python version of mxnet
    for VERSION in "${PYTHON_VERSIONS[@]}"; do
        PYTHON="python${VERSION}"
        echo "Testing ${PYTHON}"
        DOCKER_CMD="virtualenv -p \"/usr/bin/${PYTHON}\" ${PYTHON}; source \"${PYTHON}/bin/activate\"; cd ${WORKSPACE};"
        if [[ "${DEV}" == *"cpu"* ]]; then
            DOCKER_CMD="${DOCKER_CMD} pip install mxnet; python tests/python/train/test_conv.py"
        elif [[ "${DEV}" == *"cu75"* ]]; then
            DOCKER_CMD="${DOCKER_CMD} pip install mxnet-cu75; python tests/python/train/test_conv.py --gpu"
        elif [[ "${DEV}" == *"cu80"* ]]; then
            DOCKER_CMD="${DOCKER_CMD} pip install mxnet-cu80; python tests/python/train/test_conv.py --gpu"
        fi
	
        ${DOCKER_BINARY} run --rm -v ${WORKSPACE}:${WORKSPACE} ${DOCKER_TAG} bash -c "${DOCKER_CMD}"
    done

done
