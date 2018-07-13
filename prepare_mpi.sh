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

HOME_MPI_DIR=$1

if [ ! -d "$HOME_MPI_DIR" ]; then
    mkdir -p $HOME_MPI_DIR
fi

MXNET_ROOTDIR=`dirname $0`
# Default MPI Vars
DEF_MPI_URL=http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
DEF_MPI_TAR=mpich-3.2.1.tar.gz
DEF_MPI_DIR=mpich-3.2.1
DEF_MPI_BUILD=$HOME_MPI_DIR/build
DEF_MPI_LIB=$DEF_MPI_BUILD/lib/libmpi.so

if [ -e "$DEF_MPI_LIB" ]; then
    echo "${DEF_MPI_BUILD}"
    exit 0
fi

mkdir -p $DEF_MPI_BUILD
##########################
# Download MPI
##########################
echo "Downloading mpi ..." >&2
cd $HOME_MPI_DIR && wget $DEF_MPI_URL && tar xvf $DEF_MPI_TAR >&2

##########################
# Build and Install MPI
##########################
echo "Congiure & Build & Install mpi ..." >&2
cd $HOME_MPI_DIR/$DEF_MPI_DIR
./configure --prefix=$DEF_MPI_BUILD >&2
make -j >&2
make install >&2

cd $MXNET_ROOTDIR

### Return MPI_ROOT
echo "${DEF_MPI_BUILD}"

