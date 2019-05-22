#!/bin/bash

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
# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex

NOSE_COVERAGE_ARGUMENTS="--with-coverage --cover-inclusive --cover-xml --cover-branches --cover-package=mxnet"
NOSE_TIMER_ARGUMENTS="--with-timer --timer-ok 1 --timer-warning 15 --timer-filter warning,error"

clean_repo() {
    set -ex
    git clean -xfd
    git submodule foreach --recursive git clean -xfd
    git reset --hard
    git submodule foreach --recursive git reset --hard
    git submodule update --init --recursive
}

scala_prepare() {
    # Clean up maven logs
    export MAVEN_OPTS="-Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn"
}

# NOTE: There are some problems enabling ccache for v1.4.x
# in order to make it work, we need to backport this change:
# https://github.com/apache/incubator-mxnet/pull/13456/files#diff-2cb854804abadb8f92b547c4343a2aedR47
# I won't be doing it now due to time constraints - and there might be an argument
# to make a bigger PR updating CI on branch v1.4.x
# I'll keep it enabled for the 'master branch'
build_ccache_wrappers() {
    set -ex

    # Only enable ccache for master branch for now
    if [[ ${MXNET_BRANCH} != "master" ]]; then
        return
    fi

    if [ -z ${CC+x} ]; then
        echo "No \$CC set, defaulting to gcc";
        export CC=gcc
    fi
     if [ -z ${CXX+x} ]; then
       echo "No \$CXX set, defaulting to g++";
       export CXX=g++
    fi

    # Recommended by CCache: https://ccache.samba.org/manual.html#_run_modes
    # Add to the beginning of path to ensure this redirection is picked up instead
    # of the original ones. Especially CUDA/NVCC appends itself to the beginning of the
    # path and thus this redirect is ignored. This change fixes this problem
    # This hacky approach with symbolic links is required because underlying build
    # systems of our submodules ignore our CMake settings. If they use Makefile,
    # we can't influence them at all in general and NVCC also prefers to hardcode their
    # compiler instead of respecting the settings. Thus, we take this brutal approach
    # and just redirect everything of this installer has been called.
    # In future, we could do these links during image build time of the container.
    # But in the beginning, we'll make this opt-in. In future, loads of processes like
    # the scala make step or numpy compilation and other pip package generations
    # could be heavily sped up by using ccache as well.
    mkdir /tmp/ccache-redirects
    export PATH=/tmp/ccache-redirects:$PATH
    ln -s ccache /tmp/ccache-redirects/gcc
    ln -s ccache /tmp/ccache-redirects/gcc-8
    ln -s ccache /tmp/ccache-redirects/g++
    ln -s ccache /tmp/ccache-redirects/g++-8
    ln -s ccache /tmp/ccache-redirects/nvcc
    ln -s ccache /tmp/ccache-redirects/clang++-3.9
    ln -s ccache /tmp/ccache-redirects/clang-3.9
    ln -s ccache /tmp/ccache-redirects/clang++-5.0
    ln -s ccache /tmp/ccache-redirects/clang-5.0
    ln -s ccache /tmp/ccache-redirects/clang++-6.0
    ln -s ccache /tmp/ccache-redirects/clang-6.0
    ln -s ccache /usr/local/bin/gcc
    ln -s ccache /usr/local/bin/gcc-8
    ln -s ccache /usr/local/bin/g++
    ln -s ccache /usr/local/bin/g++-8
    ln -s ccache /usr/local/bin/nvcc
    ln -s ccache /usr/local/bin/clang++-3.9
    ln -s ccache /usr/local/bin/clang-3.9
    ln -s ccache /usr/local/bin/clang++-5.0
    ln -s ccache /usr/local/bin/clang-5.0
    ln -s ccache /usr/local/bin/clang++-6.0
    ln -s ccache /usr/local/bin/clang-6.0

    export NVCC=ccache

    # Uncomment if you would like to debug CCache hit rates.
    # You can monitor using tail -f ccache-log
    # export CCACHE_LOGFILE=/work/mxnet/ccache-log
    # export CCACHE_DEBUG=1
}

setup_licenses() {
    mkdir -p licenses

    cp tools/dependencies/LICENSE.binary.dependencies licenses/
    cp NOTICE licenses/
    cp LICENSE licenses/
    cp DISCLAIMER licenses/
}

build_ubuntu_cpu() {
    set -ex

    setup_licenses
    build_ccache_wrappers

    make  \
        DEV=0                         \
        ENABLE_TESTCOVERAGE=0         \
        USE_CPP_PACKAGE=0             \
        USE_MKLDNN=0                  \
        USE_BLAS=openblas             \
        USE_SIGNAL_HANDLER=1          \
        -j$(nproc)
}

build_ubuntu_cpu_mkldnn() {
    set -ex

    setup_licenses
    build_ccache_wrappers

    make  \
        DEV=0                         \
        ENABLE_TESTCOVERAGE=0         \
        USE_CPP_PACKAGE=0             \
        USE_MKLDNN=1                  \
        USE_BLAS=openblas             \
        USE_SIGNAL_HANDLER=1          \
        -j$(nproc)
}

build_ubuntu_gpu() {
    set -ex
    # unfortunately this build has problems in 3rdparty dependencies with ccache and make
    # build_ccache_wrappers

    setup_licenses

    make \
        DEV=0                                     \
        ENABLE_TESTCOVERAGE=0                     \
        USE_BLAS=openblas                         \
        USE_MKLDNN=0                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_CPP_PACKAGE=0                         \
        USE_DIST_KVSTORE=1                        \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
}

build_ubuntu_gpu_mkldnn() {
    set -ex
    # unfortunately this build has problems in 3rdparty dependencies with ccache and make
    # build_ccache_wrappers

    setup_licenses
    
    make \
        DEV=0                                     \
        ENABLE_TESTCOVERAGE=0                     \
        USE_BLAS=openblas                         \
        USE_MKLDNN=1                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_CPP_PACKAGE=0                         \
        USE_DIST_KVSTORE=1                        \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
}

# Compiles the dynamic mxnet library
# Parameters:
# $1 -> mxnet_variant: the mxnet variant to build, e.g. cpu, cu100, cu92mkl, etc.
build_ubuntu() {
    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}
    if [[ ${mxnet_variant} = "cpu" ]]; then
        build_ubuntu_cpu
    elif [[ ${mxnet_variant} = "mkl" ]]; then
        build_ubuntu_cpu_mkldnn
    elif [[ ${mxnet_variant} =~ cu[0-9]+$ ]]; then
        build_ubuntu_gpu
    elif [[ ${mxnet_variant} =~ cu[0-9]+mkl$ ]]; then
        build_ubuntu_gpu_mkldnn
    else
        echo "Error: Unrecognized mxnet variant '${mxnet_variant}'"
    fi
}

# Tests libmxnet
# Parameters:
# $1 -> mxnet_variant: The variant of the libmxnet.so library
# $2 -> python_cmd: The python command to use to execute the tests, python or python3
unittest_ubuntu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0

    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}
    local python_cmd=${2:?"This function requires a python command as the first argument"}
    
    local nose_cmd="nosetests-3.4"

    if [[ ${python_cmd} = "python" ]]; then
        nose_cmd="nosetests-2.7"
    fi

    $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/unittest
    $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/quantization    

    # https://github.com/apache/incubator-mxnet/issues/11801
    # if [[ ${mxnet_variant} = "cpu" ]] || [[ ${mxnet_variant} = "mkl" ]]; then
        # integrationtest_ubuntu_cpu_dist_kvstore
    # fi

    if [[ ${mxnet_variant} = cu* ]]; then
        $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/gpu
        $python_cmd example/image-classification/test_score.py
        integrationtest_ubuntu_gpu_dist_kvstore
    fi

    if [[ ${mxnet_variant} = *mkl ]]; then
        # skipping python 2 testing
        # https://github.com/apache/incubator-mxnet/issues/14675
        if [[ ${python_cmd} = "python3" ]]; then 
            $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/mkl
        fi
    fi
}

# quantization gpu currently only runs on P3 instances
unittest_ubuntu_python2_quantization_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-2.7 $NOSE_TIMER_ARGUMENTS --verbose tests/python/quantization_gpu
}

# quantization gpu currently only runs on P3 instances
unittest_ubuntu_python3_quantization_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 $NOSE_TIMER_ARGUMENTS --verbose tests/python/quantization_gpu
}

integrationtest_ubuntu_cpu_dist_kvstore() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_USE_OPERATOR_TUNING=0
    cd tests/nightly/
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=gluon_step_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=gluon_sparse_step_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=invalid_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=gluon_type_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --no-multiprecision
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=compressed_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=compressed_cpu --no-multiprecision
    ../../tools/launch.py -n 3 --launcher local python test_server_profiling.py
}

integrationtest_ubuntu_gpu_dist_kvstore() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    cd tests/nightly/
    ../../tools/launch.py -n 7 --launcher local python dist_device_sync_kvstore.py
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=init_gpu
    cd ../../
}

build_static_python() {
    set -ex
    pushd .
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    build_ccache_wrappers
    ./ci/cd/python/pypi/build.sh ${mxnet_variant}
    popd
}

package_static_python() {
    set -ex
    pushd .
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    ./ci/cd/python/pypi/package.sh ${mxnet_variant}
    popd
}

integration_test_python() {
    set -ex
    local python_cmd=${1:?"This function requires a python command as the first argument"}
    local gpu_enabled=${2:-"false"}

    local test_conv_params=''
    local mnist_params=''

    local pip_cmd='pip'

    if [ "${gpu_enabled}" = "true" ]; then
        mnist_params="--gpu 0"
        test_conv_params="--gpu"
    fi

    if [ "${python_cmd}" = "python3" ]; then
        pip_cmd='pip3'
    fi

    # install mxnet wheel package
    ${pip_cmd} install --user ./wheel_build/dist/*.whl

    # execute tests
    ${python_cmd} /work/mxnet/tests/python/train/test_conv.py ${test_conv_params}
    ${python_cmd} /work/mxnet/example/image-classification/train_mnist.py ${mnist_params}
}

##############################################################
# MAIN
#
# Run function passed as argument
set +x
if [ $# -gt 0 ]
then
    $@
else
    cat<<EOF

$0: Execute a function by passing it as an argument to the script:

Possible commands:

EOF
    declare -F | cut -d' ' -f3
    echo
fi
