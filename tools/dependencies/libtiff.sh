#!/usr/bin/env bash

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

# This script builds the static library of libtiff that can be used as dependency of mxnet/opencv.
set -ex
TIFF_VERSION="4.0.9"
if [[ ! -f $DEPS_PATH/lib/libtiff.a ]]; then
    # download and build libtiff
    >&2 echo "Building libtiff..."
    download \
        https://download.osgeo.org/libtiff/tiff-${TIFF_VERSION}.zip \
        ${DEPS_PATH}/libtiff.zip
    unzip -q $DEPS_PATH/libtiff.zip -d $DEPS_PATH
    pushd .
    cd $DEPS_PATH/tiff-$TIFF_VERSION
    ./configure --quiet --disable-shared --disable-jpeg --disable-zlib --disable-jbig --disable-lzma --prefix=$DEPS_PATH
    $MAKE
    $MAKE install
    popd
fi
