---
layout: page
title: Ubuntu Setup
action: Get Started
action_url: /get_started
permalink: /get_started/ubuntu_setup
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

# Installing MXNet on Ubuntu

The following installation instructions are for installing MXNet on computers running **Ubuntu 16.04**. Support for later versions of Ubuntu is [not yet available](#contributions).
<hr>

## Contents

* [CUDA Dependencies](#cuda-dependencies)
* [Quick Installation](#quick-installation)
    * [Python](#install-mxnet-for-python)
    * [pip Packages](#pip-package-availability)
* [Build from Source](#build-mxnet-from-source)
* [Installing Language Packages](#installing-language-packages-for-mxnet)
    * [R](#install-the-mxnet-package-for-r)
    * [Julia](#install-the-mxnet-package-for-julia)
    * [Scala](#install-the-mxnet-package-for-scala)
    * [Java](#install-the-mxnet-package-for-java)
    * [Perl](#install-the-mxnet-package-for-perl)
  * [Contributions](#contributions)
  * [Next Steps](#next-steps)

<hr>

## CUDA Dependencies

If you plan to build with GPU, you need to set up the environment for CUDA and cuDNN.

First, download and install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit). CUDA 9.2 is recommended.

Then download [cuDNN 7.1.4](https://developer.nvidia.com/cudnn).

Unzip the file and change to the cuDNN root directory. Move the header and libraries to your local CUDA Toolkit folder:

```bash
    tar xvzf cudnn-9.2-linux-x64-v7.1
    sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    sudo ldconfig
```

<hr>


## Quick Installation

### Install MXNet for Python

#### Dependencies

The following scripts will install Ubuntu 16.04 dependencies for MXNet Python development.

```bash
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_core.sh
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_python.sh
sudo ./ubuntu_core.sh
sudo ./ubuntu_python.sh
```

Using the latest MXNet with CUDA 9.2 package is recommended for the fastest training speeds with MXNet.

**Recommended for training:**
```bash
pip install mxnet-cu92
```

**Recommended for inference:**
```bash
pip install mxnet-cu92mkl
```

Alternatively, you can use the table below to select the package that suits your purpose.

| MXNet Version | Basic | CUDA | MKL-DNN | CUDA/MKL-DNN |
|-|-|-|-|-|
| Latest | mxnet | mxnet-cu92 | mxnet-mkl | mxnet-cu92mkl |


#### pip Package Availability

The following table presents the pip packages that are recommended for each version of MXNet.

![pip package table](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/install/pip-packages-1.4.0.png)

To install an older version of MXNet with one of the packages in the previous table add `==` with the version you require. For example for version 1.1.0 of MXNet with CUDA 8, you would use `pip install mxnet-cu80==1.1.0`.

<hr>


## Build MXNet from Source

You can build MXNet from source, and then you have the option of installing language-specific bindings, such as Scala, Java, Julia, R or Perl. This is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. (optional) Install the supported language-specific packages for MXNet. Be sure to check that section first, as some scripts may be available to handle all of the dependencies, MXNet build, and language bindings for you. Here they are again for quick access:

* [R](#install-the-mxnet-package-for-r)
* [Julia](#install-the-mxnet-package-for-julia)
* [Scala](#install-the-mxnet-package-for-scala)
* [Java](#install-the-mxnet-package-for-java)
* [Perl](#install-the-mxnet-package-for-perl)

**Note:** To change the compilation options for your build, edit the ```make/config.mk``` file prior to building MXNet. More information on this is mentioned in the different language package instructions.

### Build the Shared Library

#### Quick MXNet Build
You can quickly build MXNet from source with the following script found in the `/docs/install` folder:

```bash
cd docs/install
./install_mxnet_ubuntu_python.sh
```

Or you can go through a manual process described next.

#### Manual MXNet Installation

It is recommended that you review the general [build from source](build_from_source) instructions before continuing.

On Ubuntu versions 16.04 or later, you need the following dependencies:

**Step 1:** Install prerequisite packages.
```bash
    sudo apt-get update
    sudo apt-get install -y build-essential git ninja-build ccache
```

**For Ubuntu 18.04 and CUDA builds you need to update CMake**

```bash
#!/usr/bin/env bash
set -exuo pipefail
sudo apt remove --purge --auto-remove cmake

# Update CMAKE for correct cuda autotedetection: https://github.com/clab/dynet/issues/1457
version=3.14
build=0
mkdir -p ~/tmp
cd ~/tmp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/
./bootstrap
make -j$(nproc)
sudo make install
```


**Step 2:** Install a Math Library.

Details on the different math libraries are found in the build from source guide's [Math Library Selection](build_from_source#math-library-selection) section.

For OpenBLAS use:

```bash
    sudo apt-get install -y libopenblas-dev
```

For other libraries, visit the [Math Library Selection](build_from_source#math-library-selection) section.

**Step 3:** Install OpenCV.

*MXNet* uses [OpenCV](http://opencv.org/) for efficient image loading and augmentation operations.

```bash
    sudo apt-get install -y libopencv-dev
```

**Step 4:** Download MXNet sources and build MXNet core shared library.

If building on CPU and using OpenBLAS:

Clone the repository:

```bash
    git clone --recursive https://github.com/apache/incubator-mxnet.git
    cd incubator-mxnet
```

Build with CMake and ninja, without GPU and without MKL.

```bash
    rm -rf build
    mkdir -p build && cd build
    cmake -GNinja \
        -DUSE_CUDA=OFF \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
    ..
    ninja
```

If building on CPU and using MKL and MKL-DNN (make sure MKL is installed according to [Math Library Selection](build_from_source#math-library-selection) and [MKL-DNN README](https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_readme.html)):

```bash
    rm -rf build
    mkdir -p build && cd build
    cmake -GNinja \
        -DUSE_CUDA=OFF \
        -DUSE_MKL_IF_AVAILABLE=ON \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
    ..
    ninja
```

If building on GPU (make sure you have installed the [CUDA dependencies first](#cuda-dependencies)):
Cuda 10.1 in Ubuntu 18.04 builds fine but is not currently tested in CI.

```bash
    rm -rf build
    mkdir -p build && cd build
    cmake -GNinja \
        -DUSE_CUDA=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
    ..
    ninja
```

*Note* - You can explore and use more compilation options as they are delcared in the top of `CMakeLists.txt` and also review common [usage examples](build_from_source#usage-examples).
Optionally, you can also use a higher level, scripted version of the above with an editable CMake options file by doing the
following:

```bash
cp cmake/cmake_options.yml .
# Edit cmake_options.yml in the MXNet root to your taste
$EDITOR cmake_options.yml
# Launch a local CMake build
./dev_menu.py build
```

Building from source creates a library called ```libmxnet.so``` in the `build` folder in your MXNet project root.

After building the MXNet library, you may install language bindings.

<hr>


## Installing Language Packages for MXNet

After you have installed the MXNet core library. You may install MXNet interface packages for the programming language of your choice:
- [Python](#install-mxnet-for-python)
- [C++](#install-the-mxnet-package-for-c&plus;&plus;)
- [Clojure](#install-the-mxnet-package-for-clojure)
- [Julia](#install-the-mxnet-package-for-julia)
- [Perl](#install-the-mxnet-package-for-perl)
- [R](#install-the-mxnet-package-for-r)
- [Scala](#install-the-mxnet-package-for-scala)
- [Java](#install-the-mxnet-package-for-java)

<hr>

### Install MXNet for Python

To install the MXNet Python binding navigate to the root of the MXNet folder then run the following:

```bash
$ cd python
$ pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

#### Optional Python Packages

You may optionally install ```graphviz``` library that is used for visualizing network graphs you build on MXNet. You may also install [Jupyter Notebook](http://jupyter.readthedocs.io/) which is used for running MXNet tutorials and examples.

```bash
sudo pip install graphviz==0.8.4 \
                 jupyter
```
<hr>


### Install the MXNet Package for C++

Refer to the [C++ Package setup guide](c_plus_plus).
<hr>


### Install the MXNet Package for Clojure

Refer to the [Clojure setup guide](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package).
<hr>


### Install the MXNet Package for Julia

#### Install Julia
The package available through `apt-get` is old and not compatible with the latest version of MXNet.
Fetch the latest version (1.0.3 at the time of this writing).

```bash
wget -qO julia-10.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.3-linux-x86_64.tar.gz
```

Place the extracted files somewhere like a julia folder in your home dir.

```bash
mkdir ~/julia
mv julia-10.tar.gz ~/julia
cd ~/julia
tar xvf julia-10.tar.gz
```

Test Julia.
```bash
cd julia-1.0.3/bin
julia -e 'using InteractiveUtils; versioninfo()'
```

If you're still getting the old version, remove it.
```bash
sudo apt remove julia
```

Update your PATH to have Julia's new location. Add this to your `.zshrc`, `.bashrc`, `.profile` or `.bash_profile`.
```bash
export PATH=~/julia/julia-1.0.3/bin:$PATH
```

Validate your PATH.
```bash
echo $PATH
```

Validate Julia works and is the expected version.
```bash
julia -e 'using InteractiveUtils; versioninfo()'
```

#### Setup Your MXNet-Julia Environment

**For each of the following environment variables, add the commands to your `.zshrc`, `.bashrc`, `.profile` or `.bash_profile` to make them persist.**

Create a `julia-depot` folder and environment variable.
```bash
mkdir julia-depot
export JULIA_DEPOT_PATH=$HOME/julia/julia-depot
```

To use the Julia binding with an existing `libmxnet` installation, set the `MXNET_HOME` environment variable to the MXNet source root. For example:
```bash
export MXNET_HOME=$HOME/incubator-mxnet
```

Now set the `LD_LIBRARY_PATH` environment variable to where `libmxnet.so` is found. If you can't find it, you might have skipped the building MXNet step. Go back and [build MXNet](#build-the-shared-library) first. For example:
```bash
export LD_LIBRARY_PATH=$HOME/incubator-mxnet/lib:$LD_LIBRARY_PATH
```

Verify the location of `libjemalloc.so` and set the `LD_PRELOAD` environment variable.
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:$LD_PRELOAD
```

With all of these updates, here's an example of what you might want to have in your `.zshrc`, `.bashrc`, `.profile` or `.bash_profile`.

```
export PATH=$HOME/bin:$HOME/.local/bin:$HOME/julia/julia-1.0.3/bin:$PATH
export JULIA_DEPOT_PATH=$HOME/julia/julia-depot
export MXNET_HOME=$HOME/incubator-mxnet
export LD_LIBRARY_PATH=$HOME/incubator-mxnet/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:$LD_PRELOAD
```

Install MXNet with Julia:

```bash
julia --color=yes --project=./ -e \
	  'using Pkg; \
	   Pkg.develop(PackageSpec(name="MXNet", path = joinpath(ENV["MXNET_HOME"], "julia")))'
```

For more details about installing and using MXNet with Julia, see the [MXNet Julia documentation]({{'/api/julia'|relative_url}}).
<hr>


### Install the MXNet Package for Perl

Before you build MXNet for Perl from source code, you must complete [building the shared library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Perl package:

```bash
    sudo apt-get install libmouse-perl pdl cpanminus swig libgraphviz-perl
    cpanm -q -L "${HOME}/perl5" Function::Parameters Hash::Ordered PDL::CCS

    MXNET_HOME=${PWD}
    export LD_LIBRARY_PATH=${MXNET_HOME}/lib
    export PERL5LIB=${HOME}/perl5/lib/perl5

    cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install

    cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install

    cd ${MXNET_HOME}/perl-package/AI-MXNet/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install
```
<hr>


### Install the MXNet Package for R

Building *MXNet* from source is a 2 step process.
1. Build the *MXNet* core shared library, `libmxnet.so`, from source.
2. Build the R bindings.

#### Quick MXNet-R Installation
You can quickly build MXNet-R with the following two scripts found in the `/docs/install` folder:

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet/docs/install
./install_mxnet_ubuntu_python.sh
./install_mxnet_ubuntu_r.sh
```

Or you can go through a manual process described next.

#### Manual MXNet-R Installation

**Minimum Requirements**
1. [GCC 4.8](https://gcc.gnu.org/gcc-4.8/) or later to compile C++ 11.
2. [GNU Make](https://www.gnu.org/software/make/)

<br/>

**Build the MXNet core shared library**

**Step 1** Install build tools and git.
```bash
$ sudo apt-get update
$ sudo apt-get install -y build-essential git
```

**Step 2** Install OpenBLAS.

*MXNet* uses [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [LAPACK](https://en.wikipedia.org/wiki/LAPACK) libraries for accelerated numerical computations on CPU machine. There are several flavors of BLAS/LAPACK libraries - [OpenBLAS](https://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/) and [MKL](https://software.intel.com/en-us/intel-mkl). In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
```bash
$ sudo apt-get install -y libopenblas-dev liblapack-dev
```

**Step 3** Install OpenCV.

*MXNet* uses [OpenCV](https://opencv.org/) for efficient image loading and augmentation operations.
```bash
$ sudo apt-get install -y libopencv-dev
```

**Step 4** Download MXNet sources and build MXNet core shared library. You can clone the repository as described in the following code block, or you may try the [download links](download) for your desired MXNet version.

```bash
$ git clone --recursive https://github.com/apache/incubator-mxnet
$ cd incubator-mxnet
$ echo "USE_OPENCV = 1" >> ./config.mk
$ echo "USE_BLAS = openblas" >> ./config.mk
$ make -j $(nproc)
```

*Note* - USE_OPENCV and USE_BLAS are make file flags to set compilation options to use OpenCV and BLAS library. You can explore and use more compilation options in `make/config.mk`.

<br/>

**Step 5** Make and install the MXNet-R bindings.

```bash
$ make rpkg
```
#### Verify MXNet-R Installation

You can verify your MXNet-R installation as follows:

```bash
sudo -i R
```

At the R prompt enter the following:

```r
library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.cpu())
b <- a * 2 + 1
b
```

You should see the following output:

```
     [,1] [,2] [,3]
[1,]    3    3    3
[2,]    3    3    3
> quit()
```
<hr>


### Install the MXNet Package for Scala

To use the MXNet-Scala package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Scala Setup Instructions](scala_setup).

If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial]({{'/api/scala/docs/tutorials/mxnet_scala_on_intellij'|relative_url}}) instead.
<hr>

### Install the MXNet Package for Java

To use the MXNet-Java package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Java Setup Instructions](java_setup).

If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Java on IntelliJ tutorial]({{'/api/java/docs/tutorials/mxnet_java_on_intellij'|relative_url}}) instead.
<hr>

## Contributions

You are more than welcome to contribute easy installation scripts for other operating systems and programming languages. See the [community contributions page]({{'/community/contribute'|relative_url}}) for further information.

## Next Steps

* [Tutorials]({{'/api'|relative_url}})
* [How To]({{'/api/faq/add_op_in_backend'|relative_url}})
* [Architecture]({{'/api/architecture/overview'|relative_url}})


<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.1.0/css/all.css" integrity="sha384-lKuwvrZot6UHsBSfcMvOkWwlCMgc0TaWr+30HWe3a4ltaBwTZhyTEggF5tJv8tbt" crossorigin="anonymous">
