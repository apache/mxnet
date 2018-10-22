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

#
# This disk image and kernels for virtual testing with QEMU  is generated with some manual OS
# installation steps with the scripts and documentation found in the ci/qemu/ folder.
#
# The image has a base Debian OS and MXNet runtime dependencies installed.
# The root password is empty and there's a "qemu" user without password. SSH access is enabled as
# well.
#
# See also: ci/qemu/README.md
#

REMOTE="https://s3-us-west-2.amazonaws.com/mxnet-ci-prod-slave-data"
curl -f ${REMOTE}/vda_debian_stretch.qcow2.bz2 | bunzip2 > vda.qcow2
curl -f ${REMOTE}/vmlinuz -o vmlinuz
curl -f ${REMOTE}/initrd.img -o initrd.img

