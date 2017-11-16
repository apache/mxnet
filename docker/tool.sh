#!/usr/bin/env bash
#
# Script to build, test and push a docker container
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function show_usage() {
    echo ""
    echo "Usage: $(basename $0) COMMAND LANGUAGE DEVICE"
    echo ""
    echo "   COMMAND: build or commit."
    echo "            commit needs logined in docker hub"
    echo "   LANGUAGE: the language binding to buld, e.g. python, r-lang, julia, scala or perl"
    echo "   DEVICE: targed device, e.g. cpu, or gpu"
    echo ""
}

if (( $# < 3 )); then
    show_usage
    exit -1
fi

COMMAND=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1
LANGUAGE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1
DEVICE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

DOCKERFILE_LIB="${SCRIPT_DIR}/Dockerfiles/Dockerfile.in.lib.${DEVICE}"
if [ ! -e ${DOCKERFILE_LIB} ]; then
    echo "Error DEVICE=${DEVICE}, failed to find ${DOCKERFILE_LIB}"
    show_usage
    exit 1
fi

DOCKERFILE_LANG="${SCRIPT_DIR}/Dockerfiles/Dockerfile.in.${LANGUAGE}"
if [ ! -e ${DOCKERFILE_LANG} ]; then
    echo "Error LANGUAGE=${LANGUAGE}, failed to find ${DOCKERFILE_LANG}"
    show_usage
    exit 1
fi

if [[ "${DEVICE}" == *"gpu"* ]] && [[ "{COMMAND}" == "test" ]]; then
    DOCKER_BINARY="nvidia-docker"
else
    DOCKER_BINARY="docker"
fi

DOCKER_TAG="mxnet/${LANGUAGE}"
if [ "${DEVICE}" != 'cpu' ]; then
    DOCKER_TAG="${DOCKER_TAG}:${DEVICE}"
fi
DOCKERFILE="Dockerfile.${LANGUAGE}.${DEVICE}"

# print arguments
echo "DOCKER_BINARY: ${DOCKER_BINARY}"
echo "DOCKERFILE: ${DOCKERFILE}"
echo "DOCKER_TAG: ${DOCKER_TAG}"

if [[ "${COMMAND}" == "build" ]]; then
    rm -rf ${DOCKERFILE}
    cp ${DOCKERFILE_LIB} ${DOCKERFILE}
    cat ${DOCKERFILE_LANG} >>${DOCKERFILE}
    # To remove the following error caused by opencv
    #    libdc1394 error: Failed to initialize libdc1394"
    CMD="sh -c 'ln -s /dev/null /dev/raw1394';"
    # setup scala classpath
    if [[ "${LANGUAGE}" == "scala" ]]; then
        CMD+="CLASSPATH=\${CLASSPATH}:\`ls /mxnet/scala-package/assembly/linux-x86_64-*/target/*.jar | paste -sd \":\"\` "
    fi
    echo "CMD ${CMD} bash" >>${DOCKERFILE}
    ${DOCKER_BINARY} build -t ${DOCKER_TAG} -f ${DOCKERFILE} .
elif [[ "${COMMAND}" == "push" ]]; then
    ${DOCKER_BINARY} push ${DOCKER_TAG}
else
    echo "Unknow COMMAND=${COMMAND}"
    show_usage
    exit 1
fi
