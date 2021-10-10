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

# This script downloads OneMKL
set -ex
export INTEL_MKL="2021.3.0"
if [[  (! -e /opt/intel/oneapi/mkl/) ]]; then
    >&2 echo "Downloading mkl..."

    if [[ $PLATFORM == 'darwin' ]]; then
        download \
            https://registrationcenter-download.intel.com/akdlm/irc_nas/17960/m_onemkl_p_${INTEL_MKL}.517_offline.dmg \
            ${DEPS_PATH}/m_onemkl_p_${INTEL_MKL}.517_offline.dmg
        hdiutil attach ${DEPS_PATH}/m_onemkl_p_${INTEL_MKL}.517_offline.dmg
        pushd /Volumes/m_onemkl_p_${INTEL_MKL}.517_offline/bootstrapper.app/Contents/MacOS/
        sudo ./install.sh --silent --eula accept
        popd
    elif [[ $PLATFORM == 'linux' ]]; then
        # use wget to fetch the Intel repository public key
        download \
            https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
            ${DEPS_PATH}/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        # add to your apt sources keyring so that archives signed with this key will be trusted.
        apt-key add ${DEPS_PATH}/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        # remove the public key
        rm ${DEPS_PATH}/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
        add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
        apt-get update && \
        apt install -y intel-oneapi-mkl-${INTEL_MKL} intel-oneapi-mkl-common-${INTEL_MKL} intel-oneapi-mkl-devel-${INTEL_MKL}
    else
        >&2 echo "Not available"
    fi
fi
