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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex

function install_julia() {
    local suffix=`echo $1 | sed 's/\.//'`  # 0.7 -> 07; 1.0 -> 10
    local JLBINARY="julia-$1.tar.gz"
    local JULIADIR="/work/julia$suffix"
    local JULIA="${JULIADIR}/bin/julia"

    mkdir -p $JULIADIR
    # The julia version in Ubuntu repo is too old
    # We download the tarball from the official link:
    #   https://julialang.org/downloads/
    wget -qO $JLBINARY https://julialang-s3.julialang.org/bin/linux/x64/$1/julia-$2-linux-x86_64.tar.gz
    tar xzf $JLBINARY -C $JULIADIR --strip 1
    rm $JLBINARY

    $JULIA -e 'using InteractiveUtils; versioninfo()'
}

install_julia 0.7 0.7.0
install_julia 1.0 1.0.4
