#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -ex

# variant = cpu, native, cu80, cu100, etc.
export mxnet_variant=${1:?"Please specify the mxnet variant"}

# Due to this PR: https://github.com/apache/incubator-mxnet/pull/14899
# The setup.py expects that mkldnn_version.h be present in
# mxnet-build/3rdparty/mkldnn/build/install/include
# The artifact repository stores this file in the dependencies
# and CD unpacks it to a directory called cd_misc
# Nov. 2019 Update: With v1.1, MKL-DNN is renaming to DNNL. Hence changing the prefix of file name.
if [ -f "cd_misc/dnnl_version.h" ]; then
  mkdir -p 3rdparty/mkldnn/build/install/include
  cp cd_misc/dnnl_version.h 3rdparty/mkldnn/build/install/include/.
  cp cd_misc/dnnl_config.h 3rdparty/mkldnn/build/install/include/.
fi

# Create wheel workspace
rm -rf wheel_build
mkdir wheel_build
cd wheel_build

# Setup workspace
# setup.py expects mxnet-build to be the
# mxnet directory
ln -s ../. mxnet-build

# Copy the setup.py and other package resources
cp -R ../tools/pip/* .

# Remove comment lines from pip doc files
pushd doc
for file in $(ls); do
  sed -i '/<!--/d' ${file}
done
popd

echo "Building python package with environment:"
printenv
echo "-----------------------------------------"
pip3 install --user pypandoc

# Build wheel file - placed in wheel_build/dist
python3 setup.py bdist_wheel
