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

# This is a convenience script for calling the build scripts of all dependency libraries.
# Environment variables should be set beforehand.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"


if [[ ! $PLATFORM == 'darwin' ]]; then
    source $DIR/openblas.sh
fi
source $DIR/libz.sh
source $DIR/libturbojpeg.sh
source $DIR/libpng.sh
source $DIR/libtiff.sh
source $DIR/openssl.sh
source $DIR/curl.sh
source $DIR/eigen.sh
source $DIR/opencv.sh
source $DIR/protobuf.sh
source $DIR/cityhash.sh
source $DIR/zmq.sh
source $DIR/lz4.sh
