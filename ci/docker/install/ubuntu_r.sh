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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

# Important Maintenance Instructions:
#  Align changes with installation instructions in /get_started/ubuntu_setup.md
#  Align with R install script: /docs/install/install_mxnet_ubuntu_r.sh

set -ex
cd "$(dirname "$0")"
# install libraries for mxnet's r package on ubuntu
echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list

apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9

add-apt-repository ppa:cran/libgit2

apt-get update || true
apt-get install -y --allow-unauthenticated \
    libcairo2-dev \
    libgit2-dev \
    libssh2-1-dev \
    libssl-dev \
    libxml2-dev \
    libxt-dev \
    r-base \
    r-base-dev \
    texinfo \
    texlive \
    texlive-fonts-extra 
