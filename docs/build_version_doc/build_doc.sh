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
set -e
set -x

# This script is run on a nightly basis. Refer to Job: http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet-build-site/
# Job should pass in paramters:
# web_url=https://github.com/apache/incubator-mxnet-site
# web_branch=asf-site
# release_branch=v1.1.0 (example). This needs to come from the job config

# First parameter sent by job configuration: https://github.com/apache/incubator-mxnet-site
web_url="$1"

# Second parameter sent by job configuration: asf-site
web_branch="$2"

web_folder="VersionedWeb"

local_build="latest"

git clone $web_url $web_folder
cd $web_folder
git checkout $web_branch
cd ..
mkdir "$local_build"

# Fetch tag information
tag_list_file="tag_list.txt"
cp "$web_folder/tag.txt" "$tag_list_file"
tag_list=()
while read -r line
do
    tag_list+=("$line")
done < "$tag_list_file"

# This is the first tag found in tag.txt
latest_tag=${tag_list[0]}
echo "++++ LATEST TAG found in tag.txt file is : $latest_tag ++++"

commit_id=$(git rev-parse HEAD)

# Find the current TAG in GIT
curr_tag=${TAG}
curr_tag=${curr_tag:5}

echo "++++ CURRENT TAG IN GIT is $curr_tag ++++"

# If current tag in git is newer than latest tag found in tag.txt
if [[ "$curr_tag" != 'master' ]] && [ $curr_tag != $latest_tag ]
then
    echo "++++ Found a git TAG $curr_tag newer than mxnet repo tag $latest_tag , we need to build a new release ++++"
    echo "assigning curr_tag to latest_tag"
    latest_tag=$curr_tag
fi

# Build new released tag
if [ $latest_tag != ${tag_list[0]} ]
then
    echo " ******************************************  " 
    echo " Building new release on: $latest_tag "
    echo " ******************************************  " 
    git submodule update

    # checkout the latest release tag.
    echo "++++ Checking out and building new tag $latest_tag ++++"
    git checkout tags/$latest_tag
    make docs || exit 1
    
    # Update the tag_list (tag.txt).
    ###### content of tag.txt########
    # <latest_tag_goes_here>
    # 1.0.0
    # 0.12.1
    # 0.12.0
    # 0.11.0
    echo "++++ Adding $latest_tag to the top of the $tag_list_file ++++"
    echo -e "$latest_tag\n$(cat $tag_list_file)" > "$tag_list_file"
    cat $tag_list_file
    
    tests/ci_build/ci_build.sh doc python docs/build_version_doc/AddVersion.py --file_path "docs/_build/html/" --current_version "$latest_tag"
    tests/ci_build/ci_build.sh doc python docs/build_version_doc/AddPackageLink.py \
                                          --file_path "docs/_build/html/install/index.html" --current_version "$latest_tag"

    # The following block does the following:
    # a. copies the static html that was built from new tag to a local sandbox folder.
    # b. copies the  $tag_list_file into local sandbox tag.txt        
    # c. removes .git in VersionedWeb folder
    # d. copies VersionedWeb/versions to local sandbox versions folder.
    # e. makes a new directory with the previous TAG version. N-1 version name (example current: 1.1.0, Previous: 1.0.0)       
    # f. Copies ReadMe.md to the local sandbox build.
    # g. removes the content of VersionedWeb completely.
    # f. Adds new content from local sandbox build to VersionedWeb.          
    cp -a "docs/_build/html/." "$local_build"
    cp $tag_list_file "$local_build/tag.txt"
    rm -rf "$web_folder/.git"
    cp -a "$web_folder/versions/." "$local_build/versions"
    mkdir "$local_build/versions/${tag_list[0]}"
    cp -a "$web_folder/." "$local_build/versions/${tag_list[0]}" || exit 1
    cp "$web_folder/README.md" "$local_build"
    rm -rf "$local_build/versions/${tag_list[0]}/versions"
    rm -rf "$web_folder/*"
    cp -a "$local_build/." "$web_folder"
  
    echo " ******************************************  " 
    echo " Successfully built new release $latest_tag "
    echo " ******************************************  " 
else
    # Build latest master
    echo " ********** Building Master ************ "

    make docs || exit 1

    rm -rfv $web_folder/versions/master/*
    cp -a "docs/_build/html/." "$web_folder/versions/master"
    tests/ci_build/ci_build.sh doc python docs/build_version_doc/AddVersion.py --file_path "$web_folder/versions/master"
fi

# Update version list for all previous version website
if [ $latest_tag != ${tag_list[0]} ]
then
    total=${#tag_list[*]}
    for (( i=0; i<=$(( $total - 1 )); i++ ))
    
    do
        tests/ci_build/ci_build.sh doc python docs/build_version_doc/AddVersion.py --file_path "$web_folder/versions/${tag_list[$i]}" \
                                              --current_version "${tag_list[$i]}"
    done

    # Update master version dropdown
    tests/ci_build/ci_build.sh doc python docs/build_version_doc/AddVersion.py --file_path "$web_folder/versions/master" 
fi
