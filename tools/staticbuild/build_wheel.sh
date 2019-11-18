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

# This script builds the wheel for binary distribution and performs sanity check.
# To be used only on vanilla Ubuntu-14.04 to ensure backwards compatibility
# of all dependencies of MXNet.

set -e

#bash tools/staticbuild/install_prereq.sh
bash ci/docker/install/ubuntu_publish.sh

cp tools/pip/MANIFEST.in python/
cp -r tools/pip/doc python/

bash tools/staticbuild/build.sh $1 $2

echo $(git rev-parse HEAD) >> python/mxnet/COMMIT_HASH
cd python/

# Make wheel for testing
cp $SRC/tools/pip/setup.py pip_setup.py
python pip_setup.py bdist_wheel
rm pip_setup.py

#wheel_name=$(ls -t dist | head -n 1)
#pip install -U --user --force-reinstall dist/$wheel_name
