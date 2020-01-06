---
layout: page
title: CentOS setup
action: Get Started
action_url: /get_started
permalink: /get_started/centos_setup
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


# Installing MXNet on CentOS and other non-Ubuntu Linux systems

Step 1. Install build tools and git on `CentOS >= 7` and `Fedora >= 19`:

```bash
sudo yum groupinstall -y "Development Tools" && sudo yum install -y git
```

Step 2. Install Atlas:

```bash
sudo yum install atlas-devel
```

Installing both `git` and `cmake` or `make` by following instructions on the websites is
straightforward. Here we provide the instructions to build `gcc-4.8` from source codes.

Step 3. Install the 32-bit `libc` with one of the following system-specific commands:

```bash
sudo apt-get install libc6-dev-i386 # In Ubuntu
sudo yum install glibc-devel.i686   # In RHEL (Red Hat Linux)
sudo yum install glibc-devel.i386   # In CentOS 5.8
sudo yum install glibc-devel.i686   # In CentOS 6/7
```

Step 4. Download and extract the `gcc` source code with the prerequisites:

```bash
wget http://mirrors.concertpass.com/gcc/releases/gcc-4.8.5/gcc-4.8.5.tar.gz
tar -zxf gcc-4.8.5.tar.gz
cd gcc-4.8.5
./contrib/download_prerequisites
```

Step 5. Build `gcc` by using 10 threads and then install to `/usr/local`

```bash
mkdir release && cd release
../configure --prefix=/usr/local --enable-languages=c,c++
make -j10
sudo make install
```

Step 6. Add the lib path to your configure file such as `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
```

Step 7. Build [OpenBLAS from source](https://github.com/xianyi/OpenBLAS#installation-from-source).

Step 8. Build OpenCV

To build OpenCV from source code, you need the [cmake](https://cmake.org) library.

* If you don't have cmake or if your version of cmake is earlier than 3.6.1, run the following commands to install a newer version of cmake:

```bash
wget https://cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.tar.gz
tar -zxvf cmake-3.6.1-Linux-x86_64.tar.gz
alias cmake="cmake-3.6.1-Linux-x86_64/bin/cmake"
```

* To download and extract the OpenCV source code, run the following commands:

```bash
wget https://codeload.github.com/opencv/opencv/zip/2.4.13
unzip 2.4.13
cd opencv-2.4.13
mkdir release
cd release/
```

* Build OpenCV. The following commands build OpenCV with 10 threads. We
disabled GPU support, which might significantly slow down an MXNet program
running on a GPU processor. It also disables 1394 which might generate a
warning. Then install it on `/usr/local`.

```bash
cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j10
sudo make install
```

* Add the lib path to your configuration such as `~/.bashrc`.

```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
```
