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

if [ -z "$1" ]
  then    
    echo "Please provide a list of version tags you wish to run. Ex : \"1.1.0 1.0.0 master\""
    exit 1
  else
    tag_list=$1
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
    if [ $tag != 'master' ]
    then
        echo "$tag" >> "$tag_file"
    fi
done

# Update the specified tags with the Versions dropdown
for tag in $tag_list; do
    # This Python script is expecting the tag_list.txt and it will use that as the entries to populate
    python AddVersion.py --root_url "$root_url" --file_path "$built/versions/$tag" --current_version "$tag" || exit 1

    if [ $tag != 'master' ]
    then 
        python AddPackageLink.py --file_path "$built/versions/master/install/index.html" \
                                                   --current_version "$tag" || exit 1
    fi

    if [ $tag == $tag_default ]
    then
        cp -a "$built/versions/$tag/." "$built"
    else
        file_loc="$built/versions/$tag"
        #rm -rf "$file_loc"
        #mkdir "$file_loc"
        #cp -a $mxnet_folder/docs/_build/html/. "$file_loc"
    fi
done
    
echo "The output of this process can be found in the VersionedWeb folder."

