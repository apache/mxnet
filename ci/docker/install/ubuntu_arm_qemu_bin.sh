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

set -exuo pipefail

REMOTE="https://s3-eu-west-1.amazonaws.com/mxnet-edge-public/qemu"
#curl -O ${REMOTE}/vda.qcow2
curl -fO ${REMOTE}/vda_02.qcow2.bz2
curl -fO ${REMOTE}/initrd.img-3.16.0-6-armmp-lpae
curl -fO ${REMOTE}/vmlinuz-3.16.0-6-armmp-lpae

bunzip2 vda_02.qcow2.bz2
mv vda_02.qcow2 vda.qcow2

