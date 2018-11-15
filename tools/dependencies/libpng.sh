#!/usr/bin/env bash

# This script builds the static library of libpng that can be used as dependency of mxnet/opencv.
PNG_VERSION=1.6.34
if [[ ! -f $DEPS_PATH/lib/libpng.a ]]; then
    # download and build libpng
    >&2 echo "Building libpng..."
    curl -s -L https://github.com/glennrp/libpng/archive/v$PNG_VERSION.zip -o $DEPS_PATH/libpng.zip
    unzip -q $DEPS_PATH/libpng.zip -d $DEPS_PATH
    mkdir -p $DEPS_PATH/libpng-$PNG_VERSION/build
    cd $DEPS_PATH/libpng-$PNG_VERSION/build
    cmake \
          -D PNG_SHARED=OFF \
          -D PNG_STATIC=ON \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH \
          -D CMAKE_C_FLAGS=-fPIC ..
    make
    make install
    mkdir -p $DEPS_PATH/include/libpng
    ln -s $DEPS_PATH/include/png.h $DEPS_PATH/include/libpng/png.h
    cd -
fi
