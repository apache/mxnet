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

clean_repo() {
    set -ex
    git clean -xfd
    git submodule foreach --recursive git clean -xfd
    git reset --hard
    git submodule foreach --recursive git reset --hard
    git submodule update --init --recursive
}

build_ccache_wrappers() {
    set -ex

    rm -f cc
    rm -f cxx

    touch cc
    touch cxx

    if [ -z ${CC+x} ]; then
        echo "No \$CC set, defaulting to gcc";
        export CC=gcc
    fi

    if [ -z ${CXX+x} ]; then
       echo "No \$CXX set, defaulting to g++";
       export CXX=g++
    fi

    # this function is nessesary for cuda enabled make based builds, since nvcc needs just an executable for -ccbin

    echo -e "#!/bin/sh\n/usr/local/bin/ccache ${CC} \"\$@\"\n" >> cc
    echo -e "#!/bin/sh\n/usr/local/bin/ccache ${CXX} \"\$@\"\n" >> cxx

    chmod +x cc
    chmod +x cxx

    export CC=`pwd`/cc
    export CXX=`pwd`/cxx
}

build_wheel() {

    set -ex
    pushd .

    PYTHON_DIR=${1:-/work/mxnet/python}
    BUILD_DIR=${2:-/work/build}

    # build

    export MXNET_LIBRARY_PATH=${BUILD_DIR}/libmxnet.so

    cd ${PYTHON_DIR}
    python setup.py bdist_wheel --universal

    # repackage

    # Fix pathing issues in the wheel.  We need to move libmxnet.so from the data folder to the
    # mxnet folder, then repackage the wheel.
    WHEEL=`readlink -f dist/*.whl`
    TMPDIR=`mktemp -d`
    unzip -d ${TMPDIR} ${WHEEL}
    rm ${WHEEL}
    cd ${TMPDIR}
    mv *.data/data/mxnet/libmxnet.so mxnet
    zip -r ${WHEEL} .
    cp ${WHEEL} ${BUILD_DIR}
    rm -rf ${TMPDIR}

    popd
}

build_ubuntu_gpu_make() {
    set -ex
    make \
        USE_CUDA=1                    \
        USE_CUDNN=1                   \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_MKLML_MKL=0               \
        USE_MKLDNN=0                  \
        USE_DIST_KVSTORE=1            \
        -j$(nproc)
}

build_ubuntu_gpu_cmake() {
    set -ex
    cd /work/build
    cmake \
        -DUSE_CUDA=1               \
        -DUSE_CUDNN=1              \
        -DUSE_MKLML_MKL=0          \
        -DUSE_MKLDNN=0             \
        -DUSE_DIST_KVSTORE=1       \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja                   \
        /work/mxnet

    ninja -v
}


# Testing

sanity_check() {
    set -ex
    tools/license_header.py check
    make cpplint rcpplint jnilint
    make pylint
    nosetests-3.4 tests/tutorials/test_sanity_tutorials.py
}

unittest_ubuntu_python3_gpu() {
    set -ex
    export PYTHONPATH=./python/
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

build_docs() {
    set -ex
    pushd .
    cd /work/mxnet/docs/build_version_doc
    # Parameters are set in the Jenkins pipeline: restricted-website-build
    # $1 is the list of branches to build; $2 is the list of tags to display
    # So you can build from the 1.2.0 branch, but display 1.2.1 on the site
    ./build_all_version.sh $1 $2
    # $3 is the default version tag for the website; $4 is the base URL
    ./update_all_version.sh $2 $3 $4
    cd VersionedWeb
    tar -zcvf ../artifacts.tgz .
    popd
}


#Runs Apache RAT Check on MXNet Source for License Headers
nightly_test_rat_check() {
    set -e
    pushd .

    cd /work/deps/trunk/apache-rat/target

    # Use shell number 5 to duplicate the log output. It get sprinted and stored in $OUTPUT at the same time https://stackoverflow.com/a/12451419
    exec 5>&1
    OUTPUT=$(java -jar apache-rat-0.13-SNAPSHOT.jar -E /work/mxnet/tests/nightly/apache_rat_license_check/rat-excludes -d /work/mxnet|tee >(cat - >&5))
    ERROR_MESSAGE="Printing headers for text files without a valid license header"


    echo "-------Process The Output-------"

    if [[ $OUTPUT =~ $ERROR_MESSAGE ]]; then
        echo "ERROR: RAT Check detected files with unknown licenses. Please fix and run test again!";
        exit 1
    else
        echo "SUCCESS: There are no files with an Unknown License.";
    fi
    popd
}

# Deploy

deploy_docs() {
    set -ex
    pushd .

    make docs

    popd
}

# broken_link_checker

broken_link_checker() {
    set -ex
    ./tests/nightly/broken_link_checker_test/broken_link_checker.sh
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
