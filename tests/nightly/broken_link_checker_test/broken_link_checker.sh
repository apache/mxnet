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

#Author: Amol Lele

#software-properties-common, curl are installed in the docker container 'ubuntu_cpu'
# install git-core

#rm -rf incubator-mxnet-site && rm -rf website-link-checker

git config --global user.email \"$APACHE_USERNAME@gl.com\" && git config --global user.name \"$APACHE_USERNAME\"

echo "clone the repo and checkout the correct branch"
git clone https://$APACHE_USERNAME:$APACHE_PASSWORD@github.com/leleamol/incubator-mxnet-site.git
cd incubator-mxnet-site
git checkout link-checker
cd _urlList
echo `pwd`
./find_broken_link.sh
./check_regression.sh
cd ../.

echo "Commit the new urls found"
#git add ./_urlList/url_list.txt
#git commit -m "New Urls"
#git push origin link-checker