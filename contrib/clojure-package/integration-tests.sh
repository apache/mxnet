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

set -evx

MXNET_HOME=${PWD}
cd ${MXNET_HOME}/contrib/clojure-package
# first build the package and install it
lein install

# then run through the examples 
EXAMPLES_HOME=${MXNET_HOME}/contrib/clojure-package/examples
# use AWK pattern for blacklisting
TEST_CASES=`find ${EXAMPLES_HOME} -name test | awk '!/dontselect1|dontselect2/'`
for i in $TEST_CASES ; do
 cd ${i} && lein test
done