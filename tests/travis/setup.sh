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


if ! tests/travis/is_core_changed.sh
then
  exit 0
fi

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew update
    brew tap homebrew/science
    brew install opencv
    brew install python3
    brew install fftw
    brew install libpng
    brew install ImageMagick
    brew install swig
    if [ ${TASK} == "python_test" ]; then
        python -m pip install --user nose numpy cython scipy
        python3 -m pip install --user nose numpy cython scipy
    fi
fi

if [ ${TASK} == "lint" ]; then
    pip install --user cpplint 'pylint==1.4.4' 'astroid==1.3.6'
fi

if [ ${TASK} == "julia" ]; then
  mkdir -p ~/julia
  curl -s -L --retry 7 "https://s3.amazonaws.com/julialang/bin/linux/x64/${JULIA_VER}/julia-${JULIA_VER}-latest-linux-x86_64.tar.gz" | tar -C ~/julia -x -z --strip-components=1 -f -
  export PATH="${PATH}:${HOME}/julia/bin"
  julia -e 'versioninfo()'
fi

if [ ${TASK} == "perl_test" ]; then
    if [ ${TRAVIS_OS_NAME} == "linux" ]; then
       cpanm -q -L "${HOME}/perl5" Function::Parameters Hash::Ordered
    else
       sudo sh -c 'curl -L https://cpanmin.us | perl - App::cpanminus'
       sudo cpanm -q -n PDL Mouse Function::Parameters Hash::Ordered
    fi
fi
