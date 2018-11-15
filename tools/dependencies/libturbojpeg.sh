#!/usr/bin/env bash

# This script builds the static library of libturbojpeg that can be used as dependency of
# mxnet/opencv.
TURBO_JPEG_VERSION=1.5.90
if [[ $PLATFORM == 'darwin' ]]; then
    JPEG_NASM_OPTION="-D CMAKE_ASM_NASM_COMPILER=/usr/local/bin/nasm"
fi

if [[ ! -f $DEPS_PATH/lib/libjpeg.a ]] || [[ ! -f $DEPS_PATH/lib/libturbojpeg.a ]]; then
    # download and build libjpeg
    >&2 echo "Building libjpeg-turbo..."
    curl -s -L https://github.com/libjpeg-turbo/libjpeg-turbo/archive/$TURBO_JPEG_VERSION.zip -o $DEPS_PATH/libjpeg.zip
    unzip -q $DEPS_PATH/libjpeg.zip -d $DEPS_PATH
    mkdir -p $DEPS_PATH/libjpeg-turbo-$TURBO_JPEG_VERSION/build
    cd $DEPS_PATH/libjpeg-turbo-$TURBO_JPEG_VERSION/build
    cmake \
          -G"Unix Makefiles" \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH \
          -D CMAKE_C_FLAGS=-fPIC \
          -D WITH_JAVA=FALSE \
          -D WITH_JPEG7=TRUE \
          -D WITH_JPEG8=TRUE \
          $JPEG_NASM_OPTION \
          -D ENABLE_SHARED=FALSE ..
    make
    make install
    cd -
fi
