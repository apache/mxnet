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


# Exit script with error if any errors occur
set -e

if [[ ! $1 || ! $2 || ! $3 || ! $4 || ! $5 ]];
then
    echo "USAGE: " $(basename $"0") "USER_ID USER_NAME GROUP_ID GROUP_NAME SCRIPT"
    exit 1
fi

USER_ID=$1
USER_NAME=$2
GROUP_ID=$3
GROUP_NAME=$4
SCRIPT=$5

HOME_DIR=/home/${USER_NAME}

groupadd -f -g ${GROUP_ID} ${GROUP_NAME}
useradd -m -u ${USER_ID} -g ${GROUP_NAME} ${USER_NAME}
chown -R ${USER_NAME}:${GROUP_NAME} ${HOME_DIR}
chown -R ${USER_NAME}:${GROUP_NAME} /usr/local/lib/
echo "%${GROUP_NAME}  ALL=(ALL)       NOPASSWD: ALL" >> /etc/sudoers
su -m ${USER_NAME} -c "export HOME=${HOME_DIR}; ${SCRIPT}"
