#!/usr/bin/env bash

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

#Author: Piyush Ghai

run_models() {
	echo '=========================='
	echo "Running training files and preparing models"
	echo '=========================='
	python model_backwards_compat_train.py
	echo '=========================='
}

install_mxnet() {
	version=$1
	echo "Installing MXNet "$version
	pip install mxnet==$version --user
}

echo `pwd`
cd tests/nightly/model_backwards_compatibility_check
echo `pwd`

## Fetch the latest release tags, filtering out 'rcs' and filtering out some other irrelevant ones
## This list is sorted in descending order chronologically.
## Sample output for the below git tag command is : 1.2.0 utils 1.1.0 1.0.0 0.12.1
## so from this sample we will pick up the top two : 1.2.0 and 1.1.0 and train models on them
## Now while performing inference the latest version could be 1.3.0, which will help in validating models trained
## on 1.1.0 and 1.2.0 by loading them on the latest version (1.3.0)
## Over a period of time, the model repository will grow since with every new release we
## upload models trained on newer versions as well through this script
previous_versions=($(git tag --sort=-creatordate | grep --invert-match rc))
count=0
for version in ${previous_versions[*]}
do
	# We just need to train the previous two versions. This logic can be changed later on as welll.
	if [[ "$count" -gt 1 ]]
	then
		echo "Successfully trained files for the previous two MXNet release versions"
		exit 0
	fi

	## If MXNet major version starts with a number >=1. with a wildcard match for the minor version numbers
	if [[ $version = [1-9].[0-9].[0-9] ]]
	then
		count=$((count + 1))
		#echo $version
		install_mxnet $version
		run_models
	fi
done