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

 # Python 2.7 is installed by default, install 3.6 on top
yum -y install https://repo.ius.io/ius-release-el7.rpm
yum -y install python36u

# Install PIP
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
python3.6 get-pip.py
# Restrict numpy version to < 1.19.0 due to https://github.com/apache/incubator-mxnet/issues/18600
pip3 install nose pylint 'numpy>1.16.0,<1.19.0' nose-timer requests h5py scipy==1.2.3
