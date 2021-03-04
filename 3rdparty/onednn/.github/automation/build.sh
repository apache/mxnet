#! /bin/bash

#===============================================================================
# Copyright 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --threading)
        BUILD_THREADING="$2"
        ;;
        --mode)
        BUILD_MODE="$2"
        ;;
        --source-dir)
        SORUCE_DIR="$2"
        ;;
        --build-dir)
        BUILD_DIR="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=${BUILD_MODE} -DDNNL_BUILD_FOR_CI=ON -DDNNL_WERROR=ON"

CPU_RUNTIME="NONE"
GPU_RUNTIME="NONE"

if [ "${BUILD_THREADING}" == "tbb" ]; then
    CPU_RUNTIME="TBB"
    echo "Info: Setting DNNL_CPU_RUNTIME to TBB..."
elif [ "${BUILD_THREADING}" == "omp" ]; then
    echo "Info: Setting DNNL_CPU_RUNTIME to OMP..."
    CPU_RUNTIME="OMP"
elif [ "${BUILD_THREADING}" == "ocl" ]; then
    echo "Info: Setting DNNL_CPU_RUNTIME to OMP..."
    echo "Info: Setting DNNL_GPU_RUNTIME to OCL..."
    CPU_RUNTIME="OMP"
    GPU_RUNTIME="OCL"
else 
    echo "Error unknown threading: ${BUILD_THREADING}"
    exit 1
fi

CMAKE_OPTIONS="${CMAKE_OPTIONS} -DDNNL_CPU_RUNTIME=${CPU_RUNTIME} -DDNNL_GPU_RUNTIME=${GPU_RUNTIME}"

if [ "$(uname)" == "Linux" ]; then
    MAKE_OP="-j$(grep -c processor /proc/cpuinfo)"
else
    MAKE_OP="-j$(sysctl -n hw.physicalcpu)"
fi

cd "${SORUCE_DIR}"
echo "Calling CMake with otions: ${CMAKE_OPTIONS}"
cmake . -B${BUILD_DIR} ${CMAKE_OPTIONS} && cd ${BUILD_DIR}
make -k ${MAKE_OP}

exit $?
