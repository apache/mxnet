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

set -ex

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

## Cuts the string and gives only the major version part.
## eg : 12.3.0 ---> 12
get_major_version() {
    major=$(echo $1 | cut -d. -f1)
    echo $major
}

## We read the current major version from libinfo.py file. And we extract the major version from it.
curr_mxnet_version=$(grep -w "__version__" python/mxnet/libinfo.py | grep -o '".*"' | sed 's/"//g')
## Expected in <numeric>.<numeric>.<numeric> format
if [[ $curr_mxnet_version = [[:digit:][[:digit:]]*.[[:digit:][[:digit:]]*.[[:digit:][[:digit:]]* ]]
then
    curr_major_version=$(get_major_version $curr_mxnet_version)
else
    echo "The current major version does not comply with the regex expected. Exiting here."
    exit 1
fi

echo `pwd`
cd tests/nightly/model_backwards_compatibility_check
echo `pwd`

## Fetch the latest release tags, filtering out 'rcs' and filtering out some other irrelevant ones
## This list is sorted in descending order chronologically.
## Sample output for the below git tag command is : 1.2.0 utils 1.1.0 1.0.0 0.12.1
## so from this sample, we will pick up all the versions matching with the current latest version
## Now while performing inference the latest version could be 1.4.0, which will help in validating models trained
## on 1.1.0 and 1.2.0 by loading them on the latest version (1.4.0)
## Over a period of time, the model repository will grow since with every new release we
## upload models trained on newer versions as well through this script
previous_versions=($(git tag --sort=-creatordate | grep --invert-match rc))
count=0
for version in ${previous_versions[*]}
do
	## If MXNet major version starts with a number >=1. with a wildcard match for the minor version numbers
	## Could have used a [[:digit:]]+. as well but it was not working as a traditional regex in bash.
	## so had to resort to using [[:digit:]] [[:digit:]]* to indicate multi-digit version regex match
	## Example : #previous_versions=(12.0.0 12.12.0 12.12.12 2.0.0 1.0.4 1.2.0 v.12.0.0 beta.12.0.1)
	## When passed through the regex, the output is : [12.0.0 12.12.0 12.12.12 2.0.0 1.0.4 1.2.0]
	if [[ $version = [[:digit:][[:digit:]]*.[[:digit:][[:digit:]]*.[[:digit:][[:digit:]]* ]]
	then
#	    echo $version
	    major_version=$(get_major_version $version)
	    if [ ${major_version} -eq ${curr_major_version} ]
	        then
#			echo $version
		        install_mxnet $version
		        run_models
	    fi
	fi
done
exit 0
