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


# Build Apache MXNet from Source

Building and installing Apache MXNet from source is a three-step process. First, build
the shared `libmxnet` which provides the MXNet backend, then install your
preferred language binding and finally validate that MXNet was installed
correctly by running a small example.

1. [Obtaining the source](#obtaining-the-source-code)
2. [Installing MXNet's recommended dependencies](#installing-mxnet's-recommended-dependencies)
3. [Overview of optional dependencies and optional features](#overview-of-optional-dependencies-and-optional-features)
4. [Building MXNet](#building-mxnet)
5. [Install the language API binding(s)](#installing-mxnet-language-bindings) you would like to use for MXNet.

MXNet's newest and most popular API is Gluon. Gluon is built into the Python
binding. If Python isn't your preference, you still have more options. MXNet
supports several other language bindings. Please see the [API Documentation
page](/api) for an overview of all supported languages and their APIs.


## Obtaining the source code

To obtain the source code of the latest Apache MXNet release,
please access the [Download page](/get_started/download) and download the
`.tar.gz` source archive corresponding to the release you wish to build.

Developers can also obtain the unreleased development code from the git
repository via `git clone --recursive https://github.com/apache/mxnet`

Building a MXNet 1.x release from source requires a C++11 compliant compiler.

Building the development version of MXNet or any 2.x release from source
requires a C++17 compliant compiler. The oldest compiler versions tested during
MXNet 2 development are GCC 7, Clang 6 and MSVC 2019.

## Installing MXNet's recommended dependencies
To install the build tools and recommended dependencies, please run the
following commands respectively based on your Operating System. Please see the
next section for further explanations on the set of required and optional
dependencies of MXNet.

### Debian Linux derivatives (Debian, Ubuntu, ...)
```bash
sudo apt-get update
sudo apt-get install -y build-essential git ninja-build ccache libopenblas-dev libopencv-dev cmake
```

### Red Hat Enterprise Linux derivatives (RHEL, CentOS, Fedora, ...)
```bash
sudo yum install epel-release centos-release-scl
sudo yum install git make ninja-build automake autoconf libtool protobuf-compiler protobuf-devel \
    atlas-devel openblas-devel lapack-devel opencv-devel openssl-devel zeromq-devel python3 \ 
    devtoolset-8
source /opt/rh/devtoolset-7/enable
```
Here `devtoolset-8` refers to the [Developer Toolset
8](https://www.softwarecollections.org/en/scls/rhscl/devtoolset-8/) created by
Red Hat for developers working on CentOS or Red Hat Enterprise Linux platform
and providing the GNU Compiler Collection 9.

### macOS
```bash
# Install OS X Developer Tools
xcode-select --install

# Install Homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install dependencies
brew install cmake ninja ccache opencv
```

Note: the compiler provided by Apple on macOS does not support OpenMP. To use
OpenMP on macOS you need to install for example the Clang compiler via `brew`:

```bash
brew install llvm
```

### Windows
You can use Chocolatey software management solution to install some dependencies
on Windows.

```bash
choco install python git 7zip cmake ninja opencv
```

Currently OpenBLAS is not available from Chocolatey. You may download it from
from [the OpenBLAS release page](https://github.com/xianyi/OpenBLAS/releases)
and compile from source. Set the `OpenBLAS_HOME` environment variable to point
to the OpenBLAS directory that contains the `include` and `lib` directories for
example by typing `set OpenBLAS_HOME=C:\utils\OpenBLAS`.

If you like to compile MXNet with Visual Studio compiler, please install at
least [VS2019](https://www.visualstudio.com/downloads/).

## Overview of optional dependencies and optional features

### Math Library Selection
MXNet relies on the
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic
Linear Algebra Subprograms) library for numerical computations. In addition to
BLAS, some operators in MXNet rely on the [LAPACK (Linear Algebra
Package)](https://github.com/Reference-LAPACK/lapack), an additional set of
mathematical functions.

Several BLAS and LAPACK implementations exist. Among them, MXNet is tested with:

* [Apple Accelerate](https://developer.apple.com/documentation/accelerate)
* [ATLAS](http://math-atlas.sourceforge.net/)
* [Intel MKL](https://software.intel.com/en-us/intel-mkl)
* [OpenBLAS](https://www.openblas.net/)

Apple Accelerate and MKL are proprietary. ATLAS and OpenBLAS are Open Source. If
you don't have any specific requirements, MXNet recommends OpenBLAS as it
typically outperforms ATLAS, is portable across many platforms, provides a
LAPACK implementation and has a permissive license.

Please note that since MXNet 2.0 we are forcing static link to OpenBLAS
`libopenblas.a` on non-Windows systems. In the case that the OpenBLAS library depends
on `gfortran`, be sure to install it too as a dependency. For example, on Debian
systems you can run:
```bash
sudo apt install gfortran
```
Or on Red Hat systems, run:
```bash
sudo yum install gcc-gfortran
```

### Optional GPU support

MXNet optionally supports [NVDIA CUDA and
cuDNN](https://developer.nvidia.com/cuda-downloads) for better performance on
NVidia devices. MXNet releases in general are tested with the last two major
CUDA versions available at the time of the release. For example, CUDA 9.2 and
10.2.

To compile MXNet with CUDA support, define the `USE_CUDA` option. If you compile
MXNet on a system with NVidia GPUs, the build system will automatically detect
the CUDA Architecture. If you are compiling on a system without NVidia GPUs,
please specify the `MXNET_CUDA_ARCH` option to select the CUDA Architecture and
avoid a lengthy build targeting all common CUDA Architectures. Please see the
MXNet build configuration instructions in the next step.

MXNet also supports [NCCL](https://developer.nvidia.com/nccl) - NVIDIA's
Collective Communications Library. NCCL is useful when using MXNet on multiple
GPUs that require communication. Instructions for installing NCCL are found in
the following [Build MXNet with NCCL](#build-mxnet-with-nccl) section.

To enable building MXNet with NCCL, install NCCL and define the `USE_NCCL`
option in the MXNet build configuration in the next step.

After building with NCCL, you may optionally use the tests in
`tests/python/gpu/test_nccl.py` to ensure NCCL is enabled correctly. Please
first delete the line containing `skip(reason="Test requires NCCL library
installed and enabled during build")` before running the test. In MXNet 2.x
versions, the test can be run via `pytest --verbose
tests/python/gpu/test_nccl.py`. In MXNet 1.x it is run via `python
tests/python/gpu/test_nccl.py`.

To get the best performance out of NCCL it is recommended to set environment
variable `NCCL_LAUNCH_MODE=PARALLEL` when using NCCL version 2.1 or newer.

### Optional OpenCV support

MXNet's Image Loading and Augmentation features rely on
[OpenCV](http://opencv.org/). Image Loading and Augmentation

## Building MXNet

MXNet 1.x can be built either with a classic Makefile setup or with the `cmake`
cross platform build system. Starting with MXNet 1.7, MXNet recommends using the
`cmake` cross platform build tool.

Note: The `cmake` build requires CMake 3.13 or higher. If you are running an
older version of CMake, you will see an error message like `CMake 3.13 or higher
is required. You are running version 3.10.2`. Please update CMake on your
system. You can download and install latest CMake from https://cmake.org or via
the Python package manager `pip` with `python3 -m pip install --user --upgrade
"cmake>=3.13.2"`. After installing cmake with `pip3`, it is usually available at
`~/.local/bin/cmake` or directly as `cmake`.

Please see the [`cmake configuration
files`](https://github.com/apache/mxnet/tree/v1.x/config) files for
instructions on how to configure and build MXNet with cmake.

Up to the MXNet 1.6 release, please follow the instructions in the
[`make/config.mk`](https://github.com/apache/mxnet/blob/v1.x/make/config.mk)
file on how to configure and compile MXNet. This method is supported on all 1.x
releases.

To enable the optional MXNet C++ package, please set the `USE_CPP_PACKAGE=1`
option prior to compiling. See the [C++ guide](cpp_setup) for more information.


## Installing MXNet Language Bindings
After building MXNet's shared library, you can install other language bindings.

**NOTE:** The C++ API binding must be built when you build MXNet from source. See [Build MXNet with C++]({{'/api/cpp.html'|relative_url}}).

## Installing Language Packages for MXNet

After you have installed the MXNet core library. You may install MXNet interface
packages for the programming language of your choice:
- [Python](#install-mxnet-for-python)
- [C++](#install-the-mxnet-package-for-c&plus;&plus;)
- [Clojure](#install-the-mxnet-package-for-clojure)
- [Julia](#install-the-mxnet-package-for-julia)
- [Perl](#install-the-mxnet-package-for-perl)
- [R](#install-the-mxnet-package-for-r)
- [Scala](#install-the-mxnet-package-for-scala)
- [Java](#install-the-mxnet-package-for-java)


### Install MXNet for Python

To install the MXNet Python binding navigate to the root of the MXNet folder then run the following:

```bash
python3 -m pip install --user -e ./python
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means
that if you edit the source files, these changes will be reflected in the
package installed.

You may optionally install ```graphviz``` library that is used for visualizing
network graphs you build on MXNet. You may also install [Jupyter
Notebook](http://jupyter.readthedocs.io/) which is used for running MXNet
tutorials and examples.

```bash
python3 -m pip install --user graphviz==0.8.4 jupyter
```

Please also see the [MXNet Python API](/api/python) page.

## Contributions

You are more than welcome to contribute easy installation scripts for other operating systems and programming languages.
See the [community contributions page]({{'/community/contribute'|relative_url}}) for further information.

## Next Steps

* [Tutorials]({{'/api'|relative_url}})
* [How To]({{'/api/faq/add_op_in_backend'|relative_url}})
* [Architecture]({{'/api/architecture/overview'|relative_url}})
