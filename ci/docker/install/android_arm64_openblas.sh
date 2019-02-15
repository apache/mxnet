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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex
pushd .
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) TARGET=ARMV8 ARM_SOFTFP_ABI=1 HOSTCC=gcc NOFORTRAN=1 libs

# Ideally, a simple `make install` would do the job. However, OpenBLAS fails when running `make
# install` because it tries to execute `getarch`, which is an utility that is not compiled for the
# target platform.
#
# In order to not need to install the library we set the variable OpenBLAS_HOME in the exported
# docker image.
#
# An important gotcha here to explain the reason of not doing the `make install` by hand:
#
#   given that the compiler is installed in a different path than the usual (as is the case when
#   doing crosscompilation), then the compiler uses different paths to look for the includes and
#   libraries. So it is necessary to install it into the correct directory of includes for the
#   configured compiler ($CC and $CXX).

#   For example, When querying the default include directories with the current docker image, I get:
#
#       $ $CXX -Wp,-v -x c++ - -fsyntax-only
#       clang -cc1 version 6.0.2 based upon LLVM 6.0.2svn default target x86_64-unknown-linux-gnu
#       ignoring nonexistent directory "/usr/aarch64-linux-android/bin/.
#        ./lib/gcc/aarch64-linux-android/4.9.x/../../../../include/c++/4.9.x/backward"
#       ignoring nonexistent directory "/usr/aarch64-linux-android/bin/../sysroot/usr/local/include"
#       ignoring nonexistent directory "/usr/aarch64-linux-android/bin/../sysroot/include"
#       #include "..." search starts here:
#       #include <...> search starts here:
#        /work/deps/OpenBLAS
#        /usr/aarch64-linux-android/bin/../lib/gcc/aarch64-linux-android/4.9.x/../../../.
#         ./include/c++/4.9.x
#        /usr/aarch64-linux-android/bin/../lib/gcc/aarch64-linux-android/4.9.x/../../../.
#         ./include/c++/4.9.x/aarch64-linux-android
#        /usr/aarch64-linux-android/lib64/clang/6.0.2/include
#        /usr/aarch64-linux-android/bin/../sysroot/usr/include
#       End of search list.
#
#   The directory `/usr/include` is not in that list.
#
# As you can see, it is just easier to set the OpenBLAS_HOME variable.
popd
