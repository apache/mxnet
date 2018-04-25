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

# Takes one argument:
# * tag list - space delimited list of Github tags; Example: "1.1.0 1.0.0 master"
# Example Usage:
# ./build_all_version.sh "1.1.0 1.0.0 master"

set -e
set -x

if [ -z "$1" ]
  then
    echo "Please provide a list of version tags you wish to run."
    exit 1
  else
    IFS=$';'
    tag_list=$1
    echo "Using these tags: $tag_list"
    for tag in $tag_list; do echo $tag; done
fi

mxnet_url="https://github.com/apache/incubator-mxnet.git"
mxnet_folder="apache_mxnet"
built="VersionedWeb"

if [ ! -d "$mxnet_folder" ]; then
  mkdir $mxnet_folder
  git clone $mxnet_url $mxnet_folder --recursive
fi

if [ ! -d "$built" ]; then
  mkdir $built
  mkdir "$built/versions"
fi

# Build all versions and use latest version(First version number in $tag_list) as landing page.
for tag in $tag_list; do
    cd "$mxnet_folder"
    git fetch
    if [ $tag == 'master' ]
        then
            git checkout master
            git pull
        else
            git checkout "v$tag"
    fi

    git submodule update --init --recursive || exit 1

    make clean
    cd docs
    make clean
    make html USE_OPENMP=1 || exit 1
    cd ../../
    file_loc="$built/versions/$tag"
    if [ -d "$file_loc" ] ; then
        rm -rf "$file_loc"
    fi
    mkdir "$file_loc"
    cp -a "$mxnet_folder/docs/_build/html/." "$file_loc"
done

echo "Now you may want to run update_all_version.sh to create the production layout with the versions dropdown and other per-version corrections."
