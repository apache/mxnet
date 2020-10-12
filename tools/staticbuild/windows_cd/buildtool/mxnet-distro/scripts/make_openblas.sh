#!/usr/bin/env bash

OPENBLAS_VERSION=0.2.20
if [[ ! -e $DEPS_PATH/lib/libopenblas.a ]]; then
    # download and build openblas
    >&2 echo "Building openblas..."
    # This was a workaround for the problem fixed in https://github.com/xianyi/OpenBLAS/pull/982, until
    # it's in a release, so that builds for mac can be unblocked.
    # git clone https://github.com/xianyi/OpenBLAS $DEPS_PATH/OpenBLAS -b develop
    # cd $DEPS_PATH/OpenBLAS
    # git reset --hard 3705f5675ae45f3cc662927db1955701cf73d95b

    curl -s -L https://github.com/xianyi/OpenBLAS/archive/v$OPENBLAS_VERSION.zip -o $DEPS_PATH/openblas.zip
    unzip -q $DEPS_PATH/openblas.zip -d $DEPS_PATH
    cd $DEPS_PATH/OpenBLAS-$OPENBLAS_VERSION

    make --quiet DYNAMIC_ARCH=1 NO_SHARED=1 USE_OPENMP=1 -j $NUM_PROC || exit 1;
    make PREFIX=$DEPS_PATH install;
    cd -;
    ln -s libopenblas.a $DEPS_PATH/lib/libcblas.a;
    ln -s libopenblas.a $DEPS_PATH/lib/liblapack.a;
fi
