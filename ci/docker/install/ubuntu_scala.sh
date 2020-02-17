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
cd "$(dirname "$0")"
# install libraries for mxnet's scala package on ubuntu
echo 'Installing Scala...'

# Ubuntu 14.04
if [[ $(lsb_release -r | grep 14.04) ]]; then
   add-apt-repository -y ppa:openjdk-r/ppa
fi

# All Ubuntu
apt-get update || true
apt-get install -y \
    openjdk-8-jdk \
    openjdk-8-jre \
    software-properties-common \
    scala

# Ubuntu 14.04
if [[ $(lsb_release -r | grep 14.04) ]]; then
    curl -o apache-maven-3.3.9-bin.tar.gz -L http://www.eu.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz \
        || curl -o apache-maven-3.3.9-bin.tar.gz -L https://search.maven.org/remotecontent?filepath=org/apache/maven/apache-maven/3.3.9/apache-maven-3.3.9-bin.tar.gz

    tar xzf apache-maven-3.3.9-bin.tar.gz
    mkdir /usr/local/maven
    mv apache-maven-3.3.9/ /usr/local/maven/
    update-alternatives --install /usr/bin/mvn mvn /usr/local/maven/apache-maven-3.3.9/bin/mvn 1
    update-ca-certificates -f
else
    apt-get install -y maven
fi
