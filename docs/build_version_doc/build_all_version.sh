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
# Default repo is mxnet_url="https://github.com/apache/incubator-mxnet.git"
# Default build directory is mxnet_folder="apache-mxnet"
# Takes two required arguments and one optional:
# tag list (required)- semicolon delimited list of Github tags
#   Example: "1.2.0;1.1.0;master"
# display list (required) - semicolon delimited list of what to display on website
#   Example: "1.2.1;1.1.0;master"
# NOTE: The number of tags for the two arguments must be the same.
# Repo URL (optional) - a GitHub URL that is a fork of the MXNet project
#   When this is used the build directory will be {github_username}-mxnet

# Example Usage:
#  Build the content of the 1.2.0 branch in the main repo to the 1.2.1 folder.
#   ./build_all_version.sh "1.2.0" "1.2.1"
#  Using the main project repo, map the 1.2.0 branch to output to a 1.2.1 directory; others as is:
#   ./build_all_version.sh "1.2.0;1.1.0;master" "1.2.1;1.1.0;master"
#  Using a custom branch and fork of the repo, map the branch to master,
#    map 1.2.0 branch to 1.2.1 and leave 1.1.0 in 1.1.0:
#   ./build_all_version.sh "sphinx_error_reduction;1.2.0;1.1.0" \
#   "master;1.2.1;1.1.0" https://github.com/aaronmarkham/incubator-mxnet.git

set -e
set -x

# Set OPTS to any Sphinx build options, like -W for "warnings as errors"
OPTS=

# $1 is the list of branches/tags to build
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

# $2 is the list of output folders which will be displayed on the site
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

# $3 is the GitHub project URL or fork
if [ -z "$3" ]
  then
    echo "Using the main project URL."
    mxnet_url="https://github.com/apache/incubator-mxnet.git"
    mxnet_folder="apache-mxnet"
  else
    mxnet_url=$3
    fork=${mxnet_url##"https://github.com/"}
    fork_user=${fork%%"/incubator-mxnet.git"}
    mxnet_folder=$fork_user"-mxnet"
    echo "Building with a user supplied fork: $mxnet_url"
fi

# This is the output folder
built="VersionedWeb"


function create_repo () {
  repo_folder=$1
  mxnet_url=$2
  git clone $mxnet_url $repo_folder --recursive
  echo "Adding MXNet upstream repo..."
  cd $repo_folder
  git remote add upstream https://github.com/apache/incubator-mxnet
  cd ..
}


function refresh_branches () {
  repo_folder=$1
  cd $repo_folder
  git fetch
  git fetch upstream
  cd ..
}


function checkout () {
  repo_folder=$1
  cd $repo_folder
  # Overriding configs later will cause a conflict here, so stashing...
  git stash
  # Fails to checkout if not available locally, so try upstream
  git checkout "$repo_folder" || git branch $repo_folder "upstream/$repo_folder" && git checkout "$repo_folder" || exit 1
  if [ $tag == 'master' ]; then
    git pull
    # master gets warnings as errors for Sphinx builds
    OPTS="-W"
  else
    OPTS=
  fi
  git submodule update --init --recursive
  cd ..
}


if [ ! -d "$mxnet_folder" ]; then
  mkdir $mxnet_folder
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

    # Each tag will get its own subfolder
    if [ ! -d "$tag" ]; then
      create_repo "$tag" "$mxnet_url"
    fi

    refresh_branches $tag

    checkout $tag

    # Bring over the current configurations, so we can anticipate results.
    cp ../../mxdoc.py $tag/docs/
    cp ../../settings.ini $tag/docs/
    cp ../../conf.py $tag/docs/
    cp ../../Doxyfile $tag/docs/
    cp -a ../../_static $tag/docs/

    echo "Building $tag..."
    cd $tag/docs
    make html USE_OPENMP=1 BUILD_VER=$tag SPHINXOPTS=$OPTS || exit 1
    # Navigate back to build_version_doc folder
    cd ../../../
    # Use the display tag name for the folder name
    file_loc="$built/versions/${display_arr[${key}]}"
    if [ -d "$file_loc" ] ; then
        rm -rf "$file_loc"
    fi
    mkdir "$file_loc"
    echo "Storing artifacts for $tag in $file_loc folder..."
    cp -a "$mxnet_folder/$tag/docs/_build/html/." "$file_loc"
done

echo "Now you may want to run update_all_version.sh to create the production layout with the versions dropdown and other per-version corrections."
echo "The following pattern is recommended (tags, default tag, url base):"
echo "./update_all_version.sh \"$2\" master http://mxnet.incubator.apache.org/"
