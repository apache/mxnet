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

# Install Dependencies
sudo bash ci/docker/install/ubuntu_core.sh
sudo bash ci/docker/install/ubuntu_scala.sh

# Setup Environment Variables
# MVN_DEPLOY_OS_TYPE: linux-x86_64-cpu|linux-x86_64-gpu|osx-x86_64-cpu
# export MVN_DEPLOY_OS_TYPE=linux-x86_64-cpu

# This script is used to build the base dependencies of MXNet Scala Env
# git clone --recursive https://github.com/apache/incubator-mxnet
# cd incubator-mxnet/
# git checkout origin/master -b maven
sudo apt-get install -y libssl-dev

# This part is used to build the gpg dependencies:
sudo apt-get install -y maven gnupg gnupg2 gnupg-agent
# Mitigation for Ubuntu versions before 18.04
if ! [gpg --version | grep -q "gpg (GnuPG) 2" ]; then
    sudo mv /usr/bin/gpg /usr/bin/gpg1
    sudo ln -s /usr/bin/gpg2 /usr/bin/gpg
fi

# Run python to configure keys
python3 $PWD/scala-package/dev/buildkey.py

# Updating cache
mkdir -p ~/.gnupg
echo "default-cache-ttl 14400" > ~/.gnupg/gpg-agent.conf
echo "max-cache-ttl 14400" >> ~/.gnupg/gpg-agent.conf
export GPG_TTY=$(tty)

# Build the Scala MXNet backend
bash scala-package/dev/compile-mxnet-backend.sh $MVN_DEPLOY_OS_TYPE ./

# Scala steps to deploy
make scalapkg
make scalaunittest
make scalaintegrationtest
echo "\n\n\n" | make scalarelease-dryrun
# make scaladeploy
