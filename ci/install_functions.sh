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
set -ex

# This script handles installation of build and test dependencies
# for different platforms.

#############################
# Ubuntu Dependencies:

ubuntu_install_all_deps() {
    set -ex
    ubuntu_install_core
    ubuntu_install_python
    ubuntu_install_scala
    ubuntu_install_r
    ubuntu_install_perl
    ubuntu_install_lint
}


ubuntu_install_core() {
    
}

ubuntu_install_nvidia() {

}

ubuntu_install_perl() {

}

ubuntu_install_python() {
    
}

ubuntu_install_r() {

}

ubuntu_install_scala() {

}


ubuntu_install_lint() {

    #pip install cpplint==1.3.0 pylint==1.8.2
}

ubuntu_install_clang() {

}

ubuntu_install_mklml() {

}


centos7_all_deps() {
    set -ex
    centos7_install_core
    centos7_install_python
}

centos7_install_core() {
    
}

centos7_install_python() {
   
}

install_mkml() {
    set -ex
    pushd .
    wget -nv --no-check-certificate -O /tmp/mklml.tgz https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz
    tar -zxvf /tmp/mklml.tgz && cp -rf mklml_*/* /usr/local/ && rm -rf mklml_*
    # ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/lib/gcc/x86_64-linux-gnu/5/
    popd
}


arm64_install_all_deps() {
    arm64_install_openblas
}

arm64_install_openblas() {
    
}

android_arm64_install_all_deps() {
    android_arm64_install_ndk
    android_arm64_install_openblas
}

android_arm64_install_openblas() {

}

android_arm64_install_ndk() {

}

##############################################################
# MAIN
#
# Run function passed as argument
set +x
if [ $# -gt 0 ]
then
    $@
else
    cat<<EOF

$0: Execute a function by passing it as an argument to the script:

Possible commands:

EOF
    declare -F | cut -d' ' -f3
    echo
fi
