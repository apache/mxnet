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

# This script is for locally building website for all versions
# Built files are stored in $built
# Version numbers are stored in $tag_list.
# Version numbers are ordered from latest to old and final one is master.
set -e
set -x

tag_list="1.1.0 1.0.0 0.12.1 0.12.0 0.11.0 master"

mxnet_url="https://github.com/apache/incubator-mxnet.git"
mxnet_folder="apache_mxnet"
built="VersionedWeb"

if [ ! -d "$mxnet_folder" ]; then
  mkdir $mxnet_folder
  git clone $mxnet_url $mxnet_folder --recursive
fi

rm -rf $built
mkdir $built
mkdir "$built/versions"

cd "$mxnet_folder/docs"
tag_file="tag_list.txt"

# Write all version numbers into $tag_file
for tag in $tag_list; do
    if [ $tag != 'master' ]
    then
        echo "$tag" >> "$tag_file"
    fi
done

# Build all versions and use latest version(First version number in $tag_list) as landing page.
version_num=0
for tag in $tag_list; do
    if [ $tag == 'master' ]
    then
        git checkout master
    else
        git checkout "tags/$tag"
    fi

    git submodule update || exit 1
    cd ..
    make clean
    cd docs
    make clean
    make html USE_OPENMP=1 || exit 1

    ((++version_num))
done

echo "Now you may want to run update_all_version.sh to create the production layout with the versions dropdown and other per-version corrections."
