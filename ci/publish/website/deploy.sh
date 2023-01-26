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
#
# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

# This script requires that APACHE_PASSWORD and APACHE_USERNAME are set
# environment variables. Also, artifacts must be previously uploaded to S3
# in the MXNet public bucket (mxnet-public.s3.us-east-2.amazonaws.com).

set -ex


# Managed by Jenkins; set these env vars if running locally
# export APACHE_USERNAME=
# export APACHE_PASSWORD=

# Configuration for artifacts
version=$2
api_list=("python")
jekyll_fork=ThomasDelteil


setup_mxnet_site_repo() {
   fork=$1
   if [ ! -d "mxnet-site" ]; then
     git clone https://$APACHE_USERNAME:$APACHE_PASSWORD@github.com/aaronmarkham/mxnet-site.git
   fi

   cd mxnet-site
   git checkout asf-site
   rm -rf *
   git rm -r *
   cd ..
}


setup_mxnet_site_repo()


setup_jekyll_repo() {
   fork=$1
   if [ ! -d "mxnet.io-v2" ]; then
     git clone https://github.com/$fork/mxnet.io-v2.git
   fi
}


setup_jekyll_repo() $jekyll_fork

# Copy in the main jekyll website artifacts
web_artifacts=mxnet.io-v2/release
web_dir=mxnet-site
cp -a $web_artifacts/* $web_dir


fetch_artifacts() {
    api=$1
    artifacts=https://mxnet-public.s3.us-east-2.amazonaws.com/docs/$version/$api-artifacts.tgz
    dir=mxnet-site/api/
    wget -q $artifacts
    mkdir -p $dir
    tar xf $api-artifacts.tgz -C $dir
}

# Download and untar each of the API artifacts
for i in "${api_list[@]}"
do
    fetch_artifacts $i
done

# Commit the updates
cd mxnet-site
pwd
git branch
git add .
git commit -m "Nightly build"
git push origin asf-site
# bump the site to force replication
date > date.txt
git add date.txt
git commit -m "Bump the publish timestamp."
git push origin asf-site
