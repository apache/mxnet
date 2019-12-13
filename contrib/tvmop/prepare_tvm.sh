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

#!/bin/sh

LLVM_VERSION="8.0.0"
LLVM_ROOT="http://releases.llvm.org/${LLVM_VERSION}/"
LLVM_PKG="clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu"

os=`uname`
if [ "$os" = "Linux" ] && [ "$(arch)" = "x86_64" ]; then
    DISTRIB_ID=$(cat /etc/*-release | grep DISTRIB_ID | sed 's/DISTRIB_ID=//g' | tr '[:upper:]' '[:lower:]')
    DISTRIB_RELEASE=$(cat /etc/*-release | grep DISTRIB_RELEASE | sed 's/DISTRIB_RELEASE=//g' | tr '[:upper:]' '[:lower:]')
    if [ "$DISTRIB_ID" = "ubuntu" ]; then
        LLVM_PKG=${LLVM_PKG}-${DISTRIB_ID}-${DISTRIB_RELEASE}
    else
        echo "Downloading LLVM only supports Ubuntu. Please download manually."
        exit 1
    fi
else
    echo "Cannot identify operating system. Try downloading LLVM manually."
    exit 1
fi

LLVM_URL=${LLVM_ROOT}${LLVM_PKG}.tar.xz

TVM_PATH=`dirname $0`/../../3rdparty/tvm
DST=${TVM_PATH}/build
rm -rf $DST
mkdir -p $DST
DST=`cd $DST; pwd`

if [ -x "$(command -v curl)" ]; then
    curl -L -o "${DST}/${LLVM_PKG}.tar.xz" "$LLVM_URL"
elif [ -x "$(command -v wget)" ]; then
    wget -O "${DST}/${LLVM_PKG}.tar.xz" "$LLVM_URL"
else
    echo "curl or wget not available"
    exit 1
fi

if [ \! $? ]; then
    echo "Download from $LLVM_URL to $DST failed"
    exit 1
fi

tar -xf "$DST/${LLVM_PKG}.tar.xz" -C $DST
mv $DST/${LLVM_PKG} $DST/llvm
echo "Downloaded and unpacked LLVM libraries to $DST"
