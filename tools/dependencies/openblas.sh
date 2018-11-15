#!/usr/bin/env bash

# This script builds the static library of openblas that can be used as dependency of mxnet.
OPENBLAS_VERSION=0.3.3
if [[ ! -e $DEPS_PATH/lib/libopenblas.a ]]; then
    # download and build openblas
    >&2 echo "Building openblas..."

    curl -s -L https://github.com/xianyi/OpenBLAS/archive/v$OPENBLAS_VERSION.zip -o $DEPS_PATH/openblas.zip
    unzip -q $DEPS_PATH/openblas.zip -d $DEPS_PATH
    cd $DEPS_PATH/OpenBLAS-$OPENBLAS_VERSION

    make DYNAMIC_ARCH=1 NO_SHARED=1 USE_OPENMP=1
    make PREFIX=$DEPS_PATH install
    cd -
    ln -s libopenblas.a $DEPS_PATH/lib/libcblas.a
    ln -s libopenblas.a $DEPS_PATH/lib/liblapack.a
fi
