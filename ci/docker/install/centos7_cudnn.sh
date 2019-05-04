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

# Multipackage installation does not fail in yum
CUDNN_DOWNLOAD_SUM=4e15a323f2edffa928b4574f696fc0e449a32e6bc35c9ccb03a47af26c2de3fa
curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v7.3.1/cudnn-10.0-linux-x64-v7.3.1.20.tgz -O
echo "$CUDNN_DOWNLOAD_SUM cudnn-10.0-linux-x64-v7.3.1.20.tgz" | sha256sum -c -
tar --no-same-owner -xzf cudnn-10.0-linux-x64-v7.3.1.20.tgz -C /usr/local
rm cudnn-10.0-linux-x64-v7.3.1.20.tgz
ldconfig

