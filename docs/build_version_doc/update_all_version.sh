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

# Change to your local IP for dev builds
root_url="http://mxnet.incubator.apache.org/"
root_url="http://34.229.119.204/"

tag_list="1.1.0 1.0.0 0.12.1 0.12.0 0.11.0 master"

mxnet_folder="apache_mxnet"
built="VersionedWeb"
tag_file="tag_list.txt"

cp "$mxnet_folder/docs/$tag_file" "tag_list.txt"

# Build all versions and use latest version(First version number in $tag_list) as landing page.
version_num=0
for tag in $tag_list; do
    python AddVersion.py --root_url "$root_url" --file_path "$mxnet_folder/docs/_build/html/" --current_version "$tag" || exit 1

    if [ $tag != 'master' ]
    then 
        python AddPackageLink.py --file_path "$mxnet_folder/docs/_build/html/install/index.html" \
                                                   --current_version "$tag" || exit 1
    fi

    if [ $version_num == 0 ]
    then
        cp -a "$mxnet_folder/docs/_build/html/." "$built"
    else
        file_loc="$built/versions/$tag"
        rm -rf "$file_loc"
        mkdir "$file_loc"
        cp -a $mxnet_folder/docs/_build/html/. "$file_loc"
    fi

    ((++version_num))
done
    
echo "The output of this process can be found in the VersionedWeb folder."

