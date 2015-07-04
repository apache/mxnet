#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    make lint || exit -1
fi

if [ ${TASK} == "doc" ]; then
    make doc 2>log.txt
    (cat log.txt|grep warning) && exit -1
fi

if [ ${TASK} == "build" ]; then
    echo "USE_BLAS=blas" >> config.mk
    echo "USE_CUDA=1" >> config.mk
    echo "USE_CUDNN=0" >> config.mk
    echo "CXX=g++-4.8" >> config.mk
    export CXX="g++-4.8"
    ./dmlc-core/scripts/setup_nvcc.sh $NVCC_PREFIX
    make all || exit -1
fi

if [ ${TASK} == "test" ]; then
    cd test
    make all || exit -1
    ../scripts/travis_runtest.sh || exit -1
fi

