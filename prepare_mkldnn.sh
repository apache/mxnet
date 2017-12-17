#!/bin/bash

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

# set -ex
#
# All modification made by Intel Corporation: © 2016 Intel Corporation
#
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

MXNET_ROOTDIR="$(pwd)"
MKLDNN_ROOTDIR="$MXNET_ROOTDIR/3rdparty/mkldnn/"
MKLDNN_SRCDIR="$MKLDNN_ROOTDIR/src"
MKLDNN_BUILDDIR="$MKLDNN_ROOTDIR/build"
MKLDNN_INSTALLDIR="$MKLDNN_ROOTDIR/install"

# MKLDNN install destination
HOME_MKLDNN=$1
if [ ! -z "$HOME_MKLDNN" ]; then
  mkdir -p $HOME_MKLDNN
  if [ ! -w $HOME_MKLDNN ]; then
    echo "MKLDNN install to $HOME_MKLDNN failed, please try with sudo" >&2
    exit 1
  fi
fi

if [ -z $MKLDNNROOT ]; then
if [ ! -f "$MKLDNN_INSTALLDIR/lib/libmkldnn.so" ]; then
    mkdir -p $MKLDNN_INSTALLDIR
	cd $MKLDNN_ROOTDIR
    if [ -z $MKLROOT ] && [ ! -f $MKLDNN_INSTALLDIR/include/mkl_cblas.h ]; then
        rm -rf external && cd scripts && ./prepare_mkl.sh && cd ..
        cp -a external/*/* $MKLDNN_INSTALLDIR/.
    fi 
    echo "Building MKLDNN ..." >&2
    cd $MXNET_ROOTDIR
	g++ --version >&2
    cmake $MKLDNN_ROOTDIR -DCMAKE_INSTALL_PREFIX=$MKLDNN_INSTALLDIR -B$MKLDNN_BUILDDIR
    make -C $MKLDNN_BUILDDIR -j$(cat /proc/cpuinfo | grep processor | wc -l) VERBOSE=1 >&2
    make -C $MKLDNN_BUILDDIR install
    rm -rf $MKLDNN_BUILDDIR
fi
MKLDNNROOT=$MKLDNN_INSTALLDIR
fi

if [ -z $MKLROOT ] && [ -f $MKLDNNROOT/include/mkl_cblas.h ]; then 
  MKLROOT=$MKLDNNROOT;
fi

# user specified MKLDNN install folder
if [ -d "$HOME_MKLDNN" ]; then
  # skip if user specificed MKLDNNROOT
  [ "$MKLDNNROOT" != "$HOME_MKLDNN" ] && rsync -a $MKLDNNROOT/include $MKLDNNROOT/lib $HOME_MKLDNN/.
  [ "$MKLROOT" != "$HOME_MKLDNN" ] && rsync -a $MKLROOT/include $MKLROOT/lib $HOME_MKLDNN/.
  # update ldconfig if possible
  if [ -w /etc/ld.so.conf.d ]; then
    echo "$HOME_MKLDNN/lib" > /etc/ld.so.conf.d/mxnmkldnn.conf && ldconfig
  fi
# return value to calling script (Makefile,cmake)
  echo $HOME_MKLDNN $HOME_MKLDNN
else
  echo $MKLDNNROOT $MKLROOT
fi

