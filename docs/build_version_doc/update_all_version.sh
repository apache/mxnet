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

# This script will update the html content from building
# different tags.
# It assumes you have already run build_all_version.sh for
# the tags you want to update.

# Takes three arguments:
# * tag list - space delimited list of Github tags; Example: "1.1.0 1.0.0 master"
# * default tag - which version should the site default to; Example: 1.0.0
# * root URL - for the versions dropdown to change to production or dev server; Example: http://mxnet.incubator.apache.org/

# Example Usage:
# ./update_all_version.sh "1.1.0 1.0.0 master" 1.0.0 http://mxnet.incubator.apache.org/

set -e
set -x

MASTER_SOURCE_DIR="../../docs"
STATIC_FILES_DIR="_static"
MXNET_THEME_DIR="_static/mxnet-theme"
BUILD_HTML_DIR="_build/html"

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

if [ -z "$2" ]
  then
    echo "Please pick a version to use as a default for the website. Ex: 1.1.0"
    exit 1
  else
    tag_default=$2
fi

if [ -z "$3" ]
  then
    echo "Please provide the root url for the site. Ex: http://mxnet.incubator.apache.org/"
    exit 1
  else
    root_url=$3
fi

mxnet_folder="apache_mxnet"
built="VersionedWeb"
tag_file="tag_list.txt"

if [ -f "$tag_file" ]; then
  rm $tag_file
fi

# Write all version numbers into $tag_file for AddVersion.py to use later
# Master is added by that script by default
for tag in $tag_list; do
    echo "$tag" >> "$tag_file"
done

function update_mxnet_css {
  tag=$1
  echo "Begin update fixes.."
  # All fixes are done on the master branch of mxnet-incubator repository
  # During a nightly build, these fixes will be patched to all the versions in the asf-site repository including the master folder under versions directory.
  # copy <master folder location> <version folder location>

  echo "Copying mxnet.css from master branch to all versions...."
  cp "$MASTER_SOURCE_DIR/$STATIC_FILES_DIR/mxnet.css"  "$built/versions/$tag/_static"

  echo "Update fixes complete.."
}

function update_install {
    tag=$1
    echo "Updating installation page for $1..."
    cp "artifacts/$tag.index.html" "$built/versions/$tag/install/index.html"
}


# Update the specified tags with the Versions dropdown
# Add various artifacts depending on the version

for tag in $tag_list; do
    # This Python script is expecting the tag_list.txt and it will use that as the entries to populate

    python AddVersion.py --root_url "$root_url" --file_path "$built/versions/$tag" --current_version "$tag" --tag_default "$tag_default" || exit 1

    # Patch any fixes to all versions except 0.11.0.
    # Version 0.11.0 has old theme and does not make use of the current mxnet.css
    # It also has its install page in /getting_started, so we skip updating that
    if [ $tag != '0.11.0' ]; then
        if [ -d $built/versions/$tag ]; then
            echo "The $tag is going to be updated with new css and install pages."
            update_mxnet_css $tag
            update_install $tag
        fi
    fi

    # Update all the files that are required to go into the root folder or live version
    if [ $tag == $tag_default ]
    then
        cp -a "$built/versions/$tag/." "$built"
        echo "Updating default site's installation page."
        cp "artifacts/$tag_default.index.html" "$built/install/index.html"
        echo "Copying .htaccess from default branch to root folder...."
        cp "artifacts/.htaccess"  "$built"
    else
        file_loc="$built/versions/$tag"
    fi

    # Copy the latest README.md from master
    if [ $tag == 'master' ]; then
        cd $mxnet_folder
        git checkout master
        cp README.md ../$built
        cd ..
    fi
done

echo "The output of this process can be found in the VersionedWeb folder."

