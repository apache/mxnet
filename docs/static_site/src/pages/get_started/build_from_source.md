---
layout: page
title: Building From Source
action: Get Started
action_url: /get_started
permalink: /get_started/build_from_source
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


# Build MXNet from Source

This document explains how to build MXNet from source code.

**For Java/Scala/Clojure, please follow [this guide instead](scala_setup)**

## Overview

Building from source follows this general two-step flow of building the shared library, then installing your preferred language binding. Use the following links to jump to the different sections of this guide.

1. Build the MXNet shared library, `libmxnet.so`.
    * [Clone the repository](#clone-the-mxnet-project)
    * [Prerequisites](#prerequisites)
        * [Math library selection](#math-library-selection)
        * [Install GPU software](#install-gpu-software)
        * [Install optional software](#install-optional-software)
    * [Adjust your build configuration](#build-configurations)
    * [Build MXNet](#build-mxnet)
        * [with NCCL](#build-mxnet-with-nccl) (optional)
        * [for C++](#build-mxnet-with-c++) (optional)
        * [Usage Examples](#usage-examples)
            * [systems with GPUs and Intel CPUs](#recommended-for-Systems-with-NVIDIA-GPUs-and-Intel-CPUs)
            * [GPUs with non-Intel CPUs](#recommended-for-Systems-with-Intel-CPUs)
            * [Intel CPUs](#recommended-for-Systems-with-Intel-CPUs)
            * [non-Intel CPUs](#recommended-for-Systems-with-non-Intel-CPUs)
2. [Install the language API binding(s)](#installing-mxnet-language-bindings) you would like to use for MXNet.
MXNet's newest and most popular API is Gluon. Gluon is built into the Python binding. If Python isn't your preference, you still have more options. MXNet supports several other language APIs:
    - [Python (includes Gluon)]({{'/api/python/docs/api/index.html'|relative_url}})
    - [C++]({{'/api/cpp'|relative_url}})
    - [Clojure]({{'/api/clojure'|relative_url}})
    - [Java]({{'/api/java'|relative_url}})
    - [Julia]({{'/api/julia'|relative_url}})
    - [Perl]({{'/api/perl'|relative_url}})
    - [R]({{'/api/r'|relative_url}})
    - [Scala]({{'/api/scala'|relative_url}})

<hr>

## Build Instructions by Operating System

Detailed instructions are provided per operating system. Each of these guides also covers how to install the specific [Language Bindings](#installing-mxnet-language-bindings) you require.
You may jump to those, but it is recommended that you continue reading to understand more general "build from source" options.

* [Amazon Linux / CentOS / RHEL](centos_setup)
* [macOS](osx_setup)
* [Devices](index.html?&platform=devices&language=python&environ=pip&processor=cpu)
* [Ubuntu](ubuntu_setup)
* [Windows](windows_setup)


<hr>

## Clone the MXNet Project

1. Clone or fork the MXNet project.
```bash
git clone --recursive https://github.com/apache/incubator-mxnet mxnet
cd mxnet
```

<hr>

## Prerequisites

The following sections will help you decide which specific prerequisites you need to install.

#### Math Library Selection
It is useful to consider your math library selection prior to your other prerequisites.
MXNet relies on the
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic
Linear Algebra Subprograms) library for numerical computations.
Those can be extended with [LAPACK (Linear Algebra Package)](https://github.com/Reference-LAPACK/lapack), an additional set of mathematical functions.

MXNet supports multiple mathematical backends for computations on the CPU:
* [Apple Accelerate](https://developer.apple.com/documentation/accelerate)
* [ATLAS](http://math-atlas.sourceforge.net/)
* [MKL](https://software.intel.com/en-us/intel-mkl) (MKL, MKLML)
* [MKL-DNN](https://github.com/intel/mkl-dnn)
* [OpenBLAS](https://www.openblas.net/)

The default order of choice for the libraries if found follows the path from the most
(recommended) to less performant backends.
The following lists show this order by library and `cmake` switch.

For desktop platforms (x86_64):

1. MKL-DNN (submodule) | `USE_MKLDNN`
2. MKL | `USE_MKL_IF_AVAILABLE`
3. MKLML (downloaded) | `USE_MKLML`
4. Apple Accelerate | `USE_APPLE_ACCELERATE_IF_AVAILABLE` | Mac only
5. OpenBLAS | `BLAS` | Options: Atlas, Open, MKL, Apple

Note: If `USE_MKL_IF_AVAILABLE` is set to False then MKLML and MKL-DNN will be disabled as well for configuration
backwards compatibility.

For embedded platforms (all other and if cross compiled):

1. OpenBLAS | `BLAS` | Options: Atlas, Open, MKL, Apple

You can set the BLAS library explicitly by setting the BLAS variable to:

* Atlas
* Open
* MKL
* Apple

See the [cmake/ChooseBLAS.cmake](https://github.com/apache/incubator-mxnet/blob/master/cmake/ChooseBlas.cmake) file for the options.

[Intel's MKL (Math Kernel Library)](https://software.intel.com/en-us/mkl) is one of the most powerful math libraries

It has following flavors:

* MKL is a complete math library, containing all the functionality found in ATLAS, OpenBlas and LAPACK. It is free under
community support licensing (https://software.intel.com/en-us/articles/free-mkl),
but needs to be downloaded and installed manually.

* MKLML is a subset of MKL. It contains a smaller number of functions to reduce the
size of the download and reduce the number of dynamic libraries user needs.

<!-- [Removed until #11148 is merged.] This is the most effective option since it can be downloaded and installed automatically
by the cmake script (see cmake/DownloadMKLML.cmake).-->

* MKL-DNN is a separate open-source library, it can be used separately from MKL or MKLML. It is
shipped as a subrepo with MXNet source code (see 3rdparty/mkldnn or the [MKL-DNN project](https://github.com/intel/mkl-dnn))

Since the full MKL library is almost always faster than any other BLAS library it's turned on by default,
however it needs to be downloaded and installed manually before doing `cmake` configuration.
Register and download on the [Intel performance libraries website](https://software.intel.com/en-us/performance-libraries).
You can also install MKL through [YUM](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-yum-repo)
or [APT](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo) Repository.

Note: MKL is supported only for desktop builds and the framework itself supports the following
hardware:

* Intel® Xeon Phi™ processor
* Intel® Xeon® processor
* Intel® Core™ processor family
* Intel Atom® processor

If you have a different processor you can still try to use MKL, but performance results are
unpredictable.


#### Install GPU Software

If you want to run MXNet with GPUs, you must install [NVDIA CUDA and cuDNN](https://developer.nvidia.com/cuda-downloads).


#### Install Optional Software

These might be optional, but they're typically desirable as the extend or enhance MXNet's functionality.

* [OpenCV](http://opencv.org/) - Image Loading and Augmentation. Each operating system has different packages and build from source options for OpenCV. Refer to your OS's link in the [Build Instructions by Operating System](#build-instructions-by-operating-system) section for further instructions.
* [NCCL](https://developer.nvidia.com/nccl) - NVIDIA's Collective Communications Library. Instructions for installing NCCL are found in the following [Build MXNet with NCCL](#build-mxnet-with-nccl) section.

More information on turning these features on or off are found in the following [build configurations](#build-configurations) section.


<hr>

## Build Configurations

There is a configuration file for make,
[`make/config.mk`](https://github.com/apache/incubator-mxnet/blob/master/make/config.mk), that contains all the compilation options. You can edit it and then run `make` or `cmake`. `cmake` is recommended for building MXNet (and is required to build with MKLDNN), however you may use `make` instead. For building with Java/Scala/Clojure, only `make` is supported.

**NOTE:** When certain set of build flags are set, MXNet archive increases to more than 4 GB. Since MXNet uses archive internally archive runs into a bug ("File Truncated": [bugreport](https://sourceware.org/bugzilla/show_bug.cgi?id=14625)) for archives greater than 4 GB. Please use ar version 2.27 or greater to overcome this bug. Please see https://github.com/apache/incubator-mxnet/issues/15084 for more details.

<hr>

## Build MXNet

### Build MXNet with NCCL
- Download and install the latest NCCL library from NVIDIA.
- Note the directory path in which NCCL libraries and header files are installed.
- Ensure that the installation directory contains ```lib``` and ```include``` folders.
- Ensure that the prerequisites for using NCCL such as Cuda libraries are met.
- Append the ```config.mk``` file with following, in addition to the CUDA related options.
- USE_NCCL=1
- USE_NCCL_PATH=path-to-nccl-installation-folder

``` bash
echo "USE_NCCL=1" >> make/config.mk
echo "USE_NCCL_PATH=path-to-nccl-installation-folder" >> make/config.mk
cp make/config.mk .
```
- Run make command
``` bash
make -j"$(nproc)"
```

#### Validating NCCL
- Follow the steps to install MXNet Python binding.
- Comment the following line in ```test_nccl.py``` file at ```incubator-mxnet/tests/python/gpu/test_nccl.py```
``` bash
@unittest.skip("Test requires NCCL library installed and enabled during build")
```
- Run test_nccl.py script as follows. The test should complete. It does not produce any output.
``` bash
nosetests --verbose tests/python/gpu/test_nccl.py
```

**Recommendation to get the best performance out of NCCL:**
It is recommended to set environment variable NCCL_LAUNCH_MODE to PARALLEL when using NCCL version 2.1 or newer.

<hr>

### Build MXNet with C++

* To enable C++ package, just add `USE_CPP_PACKAGE=1` when you run `make` or `cmake` (see examples).

<hr>

### Usage Examples

For example, you can specify using all cores on Linux as follows:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```


#### Recommended for Systems with NVIDIA GPUs and Intel CPUs
* Build MXNet with `cmake` and install with MKL DNN, GPU, and OpenCV support:

```bash
mkdir build && cd build
cmake -DUSE_CUDA=1 -DUSE_CUDA_PATH=/usr/local/cuda -DUSE_CUDNN=1 -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```

#### Recommended for Systems with NVIDIA GPUs
* Build with both OpenBLAS, GPU, and OpenCV support:

```bash
mkdir build && cd build
cmake -DBLAS=open -DUSE_CUDA=1 -DUSE_CUDA_PATH=/usr/local/cuda -DUSE_CUDNN=1 -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```

#### Recommended for Systems with Intel CPUs
* Build MXNet with `cmake` and install with MKL DNN, and OpenCV support:

```bash
mkdir build && cd build
cmake -DUSE_CUDA=0 -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```

#### Recommended for Systems with non-Intel CPUs
* Build MXNet with `cmake` and install with OpenBLAS and OpenCV support:

```bash
mkdir build && cd build
cmake -DUSE_CUDA=0 -DBLAS=open -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```

#### Other Examples

* Build without using OpenCV:

```bash
mkdir build && cd build
cmake -DUSE_OPENCV=0 -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```

* Build on **macOS** with the default BLAS library (Apple Accelerate) and Clang installed with `xcode` (OPENMP is disabled because it is not supported by the Apple version of Clang):

```bash
mkdir build && cd build
cmake -DBLAS=apple -DUSE_OPENCV=0 -DUSE_OPENMP=0 -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```

* To use OpenMP on **macOS** you need to install the Clang compiler, `llvm` (the one provided by Apple does not support OpenMP):

```bash
brew install llvm
mkdir build && cd build
cmake -DBLAS=apple -DUSE_OPENMP=1 -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -v
```

<hr>

## Installing MXNet Language Bindings
After building MXNet's shared library, you can install other language bindings.

**NOTE:** The C++ API binding must be built when you build MXNet from source. See [Build MXNet with C++]({{'/api/cpp'|relative_url}}).

The following table provides links to each language binding by operating system:

| Language | [Ubuntu](ubuntu_setup) | [macOS](osx_setup) | [Windows](windows_setup) |
| --- | ----  | --- | ------- |
| Python | [Ubuntu guide](ubuntu_setup.html#install-mxnet-for-python) | [OSX guide](osx_setup) | [Windows guide](windows_setup.html#install-mxnet-for-python) |
| C++ | [C++ guide](cpp_setup) | [C++ guide](cpp_setup) | [C++ guide](cpp_setup) |
| Clojure | [Clojure guide](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package) | [Clojure guide](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package) | n/a |
| Julia | [Ubuntu guide](ubuntu_setup.html#install-the-mxnet-package-for-julia) | [OSX guide](osx_setup.html#install-the-mxnet-package-for-julia) | [Windows guide](windows_setup.html#install-the-mxnet-package-for-julia) |
| Perl | [Ubuntu guide](ubuntu_setup.html#install-the-mxnet-package-for-perl) | [OSX guide](osx_setup.html#install-the-mxnet-package-for-perl) | n/a |
| R | [Ubuntu guide](ubuntu_setup.html#install-the-mxnet-package-for-r) | [OSX guide](osx_setup.html#install-the-mxnet-package-for-r) | [Windows guide](windows_setup.html#install-the-mxnet-package-for-r) |
| Scala | [Scala guide](scala_setup.html) | [Scala guide](scala_setup.html) | n/a |
| Java | [Java guide](java_setup.html) | [Java Guide](java_setup.html) | n/a |

