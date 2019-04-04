#!/usr/bin/env bash -exuo pipefail

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
rm -rf initrd
mkdir -p initrd
cd initrd
gunzip -c ../installer-initrd.gz | cpio -i
cp ../preseed.cfg .
cp ../initrd_modif/inittab etc/inittab
cp ../initrd_modif/S10syslog lib/debian-installer-startup.d/S10syslog
find .  | cpio --create --format 'newc'  | gzip -c > ../installer-initrd_automated.gz
echo "Done!"
