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

set -ex
# install libraries for mxnet's python package on ubuntu
apt-get update || true
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update || true
apt-get install -y python3.7-dev python3.7-distutils virtualenv wget
# setup symlink in /usr/local/bin to override python3 version
ln -sf /usr/bin/python3.7 /usr/local/bin/python3

# the version of the pip shipped with ubuntu may be too lower, install a recent version here
wget -nv https://bootstrap.pypa.io/get-pip.py
/usr/local/bin/python3 get-pip.py
pip3 install -r /work/requirements
