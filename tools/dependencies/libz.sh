#!/usr/bin/env bash

# This script builds the static library of libz that can be used as dependency of mxnet.
ZLIB_VERSION=1.2.6
if [[ ! -f $DEPS_PATH/lib/libz.a ]]; then
    # Download and build zlib
    >&2 echo "Building zlib..."
    curl -s -L https://github.com/LuaDist/zlib/archive/$ZLIB_VERSION.zip -o $DEPS_PATH/zlib.zip
    unzip -q $DEPS_PATH/zlib.zip -d $DEPS_PATH
    mkdir -p $DEPS_PATH/zlib-$ZLIB_VERSION/build
    cd $DEPS_PATH/zlib-$ZLIB_VERSION/build
    cmake \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH \
          -D BUILD_SHARED_LIBS=OFF ..
    make
    make install
    cd -
fi
