#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

SOURCE=$1

# get Apache Rat jar
wget http://apache.mirrors.lucidnetworks.net/creadur/apache-rat-0.12/apache-rat-0.12-bin.tar.gz
tar -xvzf apache-rat-0.12-bin.tar.gz && rm apache-rat-0.12-bin.tar.gz

# run verification on given directory
java -jar apache-rat-0.12/apache-rat-0.12.jar -E ${SOURCE}/.rat-excludes -d ${SOURCE} > rat_report.txt

if [ $? -ne 0 ]; then
   echo "RAT exited abnormally"
   exit 1
fi

# grep for files with bad licenses
echo "Grep for bad licences"
ERRORS="$(cat rat_report.txt | grep -e "!?????")"
rm rat_report.txt

if test ! -z "$ERRORS"; then
    echo "Could not find Apache license headers in the following files:"
    echo "$ERRORS"
    COUNT=`echo "${ERRORS}" | wc -l`
    exit 1 
else
    echo -e "RAT checks passed."
fi

exit 0

