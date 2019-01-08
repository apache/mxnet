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

# Setup Environment Variables
# MAVEN_PUBLISH_OS_TYPE: linux-x86_64-cpu|linux-x86_64-gpu|osx-x86_64-cpu
# export MAVEN_PUBLISH_OS_TYPE=linux-x86_64-cpu

# Run python to configure keys
python3 $PWD/scala-package/dev/buildkey.py

# Updating cache
mkdir -p ~/.gnupg
echo "default-cache-ttl 14400" > ~/.gnupg/gpg-agent.conf
echo "max-cache-ttl 14400" >> ~/.gnupg/gpg-agent.conf
echo "allow-loopback-pinentry" >> ~/.gnupg/gpg-agent.conf
echo "pinentry-mode loopback" >> ~/.gnupg/gpg-agent.conf
export GPG_TTY=$(tty)

cd scala-package
VERSION=$(mvn -q -Dexec.executable="echo" -Dexec.args='${project.version}' --non-recursive exec:exec)
cd ..

# echo "\n\n$VERSION\n" | make scalarelease-dryrun
make scaladeploy CI=1

# Clear all password .xml files, exp files, and gpg key files
rm -rf ~/.m2/*.xml ~/.m2/key.asc ~/.m2/*.exp
