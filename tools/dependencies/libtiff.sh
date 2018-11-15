#!/usr/bin/env bash

# This script builds the static library of libtiff that can be used as dependency of mxnet/opencv.
TIFF_VERSION="4-0-9"
if [[ ! -f $DEPS_PATH/lib/libtiff.a ]]; then
    # download and build libtiff
    >&2 echo "Building libtiff..."
    curl -s -L https://gitlab.com/libtiff/libtiff/-/archive/Release-v$TIFF_VERSION/libtiff-Release-v$TIFF_VERSION.zip -o $DEPS_PATH/libtiff.zip
    unzip -q $DEPS_PATH/libtiff.zip -d $DEPS_PATH
    cd $DEPS_PATH/libtiff-Release-v$TIFF_VERSION
    ./configure --quiet --disable-shared --disable-jpeg --disable-zlib --disable-jbig --disable-lzma --prefix=$DEPS_PATH
    make
    make install
    cd -
fi
