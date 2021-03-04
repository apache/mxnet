#! /bin/bash

#===============================================================================
# Copyright 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

UBUNTU_DISTRO="$(cat /etc/lsb-release | grep CODENAME | sed 's/.*=//g')"

sudo add-apt-repository "deb http://apt.llvm.org/${UBUNTU_DISTRO}/ llvm-toolchain-${UBUNTU_DISTRO}-9 main"
sudo add-apt-repository "deb-src http://apt.llvm.org/${UBUNTU_DISTRO}/ llvm-toolchain-${UBUNTU_DISTRO}-9 main"
sudo apt update && sudo apt install -y clang-9 lldb-9 lld-9 clang-format-9

sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 100
sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-9 100
sudo update-alternatives --set clang /usr/bin/clang-9
sudo update-alternatives --set clang++ /usr/bin/clang++-9
sudo update-alternatives --set clang-format /usr/bin/clang-format-9