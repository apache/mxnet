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


# this is a util script to test whether the "core" of
# mxnet has changed. Please modify the regex patterns here
# to ensure the components are covered if you add new "core"
# components to mxnet

# temporarily disable this b/c the OS X tests are failing mysteriously
exit 0

# DEBUG
echo "Files changed in this PR includes:"
echo "**********************************"
git diff --name-only HEAD^
echo "**********************************"

# we ignore examples, and docs
core_patterns=(
  '^dmlc-core'
  '^matlab'
  '^plugin'
  '^python'
  '^src'
  '^tools'
  '^R-package'
  '^amalgamation'
  '^include'
  '^mshadow'
  '^ps-lite'
  '^scala-package'
  '^tests'
)

for pat in ${core_patterns[@]}; do
  if git diff --name-only HEAD^ | grep "$pat"
  then
    exit
  fi
done

echo "I think we are good to skip this travis ci run now"
exit 1 # means nothing has changed
