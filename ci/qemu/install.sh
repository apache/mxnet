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

set -ex
rm -f vda.qcow2
sudo ./preseed.sh
qemu-img create -f qcow2 vda.qcow2 10G
qemu-system-arm -M virt -m 1024 \
  -kernel installer-vmlinuz \
  -append BOOT_DEBUG=2,DEBIAN_FRONTEND=noninteractive \
  -initrd installer-initrd_automated.gz \
  -drive if=none,file=vda.qcow2,format=qcow2,id=hd \
  -device virtio-blk-device,drive=hd \
  -netdev user,id=mynet \
  -device virtio-net-device,netdev=mynet \
  -nographic -no-reboot
