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

# This script imports the headers from eigen3 that can be used to in opencv.
set -ex
EIGEN_VERSION=3.3.4
if [[ ! -d $DEPS_PATH/include/eigen3 ]]; then
    # download eigen
    >&2 echo "Loading eigen..."
    curl -s -L https://github.com/eigenteam/eigen-git-mirror/archive/$EIGEN_VERSION.zip -o $DEPS_PATH/eigen.zip
    unzip -q $DEPS_PATH/eigen.zip -d $DEPS_PATH
    mkdir -p $DEPS_PATH/eigen-git-mirror-$EIGEN_VERSION/build
    pushd .
    cd $DEPS_PATH/eigen-git-mirror-$EIGEN_VERSION/build
    cmake \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH ..
    $MAKE install
    popd
fi
