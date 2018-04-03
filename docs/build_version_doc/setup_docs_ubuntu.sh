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


# If you need to build <= v0.12.0 then use a Python 2 environment
# mxdoc.py - a sphinx extension, was not Python 3 compatible in the old versions
# source activate mxnet_p27

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    doxygen \
    git \
    libjemalloc-dev \
    pandoc \
    software-properties-common

pip install --user \
    beautifulsoup4 \
    breathe \
    CommonMark==0.5.4 \
    h5py \
    mock==1.0.1 \
    pypandoc \
    recommonmark==0.4.0 \
    sphinx==1.5.6

# Setup scala
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install -y \
  sbt \
  scala

# Optionally setup Apache2
sudo apt-get install -y apache2
sudo ufw allow 'Apache Full'
# turn on mod_rewrite
sudo a2enmod rewrite

echo 'To enable redirects you need to edit /etc/apache2/apache2.conf '
echo '--> Change directives for Directory for /var/www/html using the following: '
echo '       AllowOverride all '
echo '--> Then restart apache with: '
echo '       sudo systemctl restart apache2'

# Cleanup
sudo apt autoremove -y
