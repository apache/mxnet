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

# Takes two arguments:
# tag list - semicolon delimited list of Github tags
#   Example: "1.2.0;1.1.0;master"
# display list - semicolon delimited list of what to display on website
#   Example: "1.2.1;1.1.0;master"
# The number of tags for the two arguments must be the same.
# Example Usage:
#   ./build_all_version.sh "1.2.0;1.1.0;master" "1.2.1;1.1.0;master"
#   ./build_all_version.sh "1.2.0" "1.2.1"

set -e
set -x

if [ -z "$1" ]
  then
    echo "Please provide a list of branches or tags you wish to build."
    exit 1
  else
    IFS=$';'
    tag_list=$1
    echo "Using these tags: $tag_list"
    build_arr=($tag_list)
fi

if [ -z "$2" ]
  then
    echo "Please provide a list of version tags you wish to display on the site."
    exit 1
  else
    IFS=$';'
    tags_to_display=$2
    echo "Displaying these tags: $tags_to_display"
    display_arr=($tags_to_display)
    for key in ${!build_arr[@]}; do
        echo "Branch/tag ${build_arr[${key}]} will be displayed as ${display_arr[${key}]}"
    done
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
  else
    if [ ! -d "$built/versions" ]; then
      mkdir "$built/versions"
    fi
fi

# Checkout each tag and build it
# Then store it in a folder according to the desired display tag
for key in ${!build_arr[@]}; do
    tag=${build_arr[${key}]}
    cd "$mxnet_folder"
    git fetch
    if [ $tag == 'master' ]
        then
            git checkout master
            git pull
            echo "Building master..."
        else
            # Use "v$tag" for branches or pass that in from jenkins
            git checkout "$tag"
            echo "Building $tag..."
    fi

    git submodule update --init --recursive || exit 1

    make clean
    cd docs
    make clean
    make html USE_OPENMP=1 || exit 1
    cd ../../
    # Use the display tag name for the folder name
    file_loc="$built/versions/${display_arr[${key}]}"
    if [ -d "$file_loc" ] ; then
        rm -rf "$file_loc"
    fi
    mkdir "$file_loc"
    echo "Storing artifacts for $tag in $file_loc folder..."
    cp -a "$mxnet_folder/docs/_build/html/." "$file_loc"
done

echo "Now you may want to run update_all_version.sh to create the production layout with the versions dropdown and other per-version corrections."
