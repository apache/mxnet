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

set -ex

echo "Install dependencies"
apt-get update || true
apt-get install -y curl subversion openjdk-8-jdk openjdk-8-jre software-properties-common
apt-add-repository ppa:webupd8team/java
apt-get update

# Installing maven 3.6.3 because default maven 3.7.x requires maven compiler >= 1.6
url="https://mirror.olnevhost.net/pub/apache/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz"
install_dir="/opt/maven"
mkdir ${install_dir}
curl -fsSL ${url} | tar zx --strip-components=1 -C ${install_dir}
cat << EOF > /etc/profile.d/maven.sh
#!/bin/sh
export M2_HOME=${install_dir}
export MAVEN_HOME=${install_dir}
export PATH=${install_dir}/bin:${PATH}
EOF
source /etc/profile.d/maven.sh
cat /etc/profile.d/maven.sh

echo "download RAT"
#svn co http://svn.apache.org/repos/asf/creadur/rat/trunk/
svn co http://svn.apache.org/repos/asf/creadur/rat/tags/apache-rat-project-0.13/

echo "cd into directory"
cd apache-rat-project-0.13

echo "mvn install"
mvn -Dmaven.test.skip=true install
