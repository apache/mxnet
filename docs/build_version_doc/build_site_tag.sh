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


# How this script works:
# 1. Receive tag list
#    Looks like: tag_list="1.1.0 1.0.0 0.12.1 0.12.0 0.11.0 master"
# 2. Receive default tag (for main website view)
# 3. Receive root URL
# 4. Call build and then update scripts

# Take user input or check env var for tag list
if [ -z "$1" ]
  then
    echo "No tag list supplied... trying environment variable $TAG_LIST"
  else
    tag_list="${TAG_LIST:-"$1"}"
    echo "Using these tags: $1"
fi

if [ -z "$tag_list" ]
  then
    echo "No tags defined"
    exit 1
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

# Pass params to build and update scripts
for tag in $tag_list; do
  ./build_all_version.sh $tag || exit 1
done

./update_all_version.sh "$tag_list" $tag_default $root_url || exit 1

