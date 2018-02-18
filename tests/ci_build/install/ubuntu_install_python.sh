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
set -e
set -x

# install libraries for mxnet's python package on ubuntu
apt-get update && apt-get install -y python-dev python3-dev

# the version of the pip shipped with ubuntu may be too lower, install a recent version here
cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && python2 get-pip.py

pip2 install nose pylint numpy nose-timer requests h5py scipy
pip3 install nose pylint numpy nose-timer requests h5py scipy
