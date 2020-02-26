---
layout: page
title: OSX Setup
action: Get Started
action_url: /get_started
permalink: /get_started/osx_setup
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

# Installing MXNet from source on OS X (Mac)

The following installation instructions are for building MXNet from source. For
instructions to build MXNet from source on other platforms, see the general
[Build From Source guide](build_from_source).

Instead of building from source, you can install a binary version of MXNet. For
that, please follow the information at [Get Started](get_started).

Building MXNet from source is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. (optional) Install the supported language-specific packages for MXNet.

If you plan to build with GPU, you need to set up the environment for CUDA and
cuDNN. Please follow the [NVIDIA CUDA Installation Guide for Mac OS
X](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) and
[cuDNN Installation
Guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-mac).
Note that CUDA stopped supporting macOS in 2019 and future versions of CUDA may
not support macOS.

## Contents

* [Build the MXNet shared library from source](#build-mxnet-from-source)
* [Install Language Packages](#installing-language-packages-for-mxnet)
    * [R](#install-the-mxnet-package-for-r)
    * [Julia](#install-the-mxnet-package-for-julia)
    * [Scala](#install-the-mxnet-package-for-scala)
    * [Java](#install-the-mxnet-package-for-java)
    * [Perl](#install-the-mxnet-package-for-perl)
  * [Contributions](#contributions)
  * [Next Steps](#next-steps)

<hr>


## Build the MXNet shared library from source

On OS X, you need the following dependencies:

**Step 1:** Install prerequisite packages.

```bash
# Install OS X Developer Tools
xcode-select --install

# Install Homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install dependencies
brew install cmake ninja ccache opencv
```

`opencv` is an optional dependency. You can delete it from above `brew install`
line and build MXNet without OpenCV support by setting `USE_OPENCV` to `OFF` in
the configuration file described below.


**Step 2:** Download MXNet sources and configure

Clone the repository:

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet
cp config/darwin.cmake config.cmake
```

Please edit the config.cmake file based on your needs. The file contains a
series of `set(name value CACHE TYPE "Description")` entries. For example, to
build without Cuda, change `set(USE_CUDA ON CACHE TYPE "Build with CUDA
support")` to `set(USE_CUDA OFF CACHE TYPE "Build with CUDA support")`.

For a GPU-enabled build make sure you have installed the [CUDA dependencies
first](#cuda-dependencies)). When building a GPU-enabled build on a machine
without GPU, MXNet build can't autodetect your GPU architecture and will target
all available GPU architectures. Please set the `MXNET_CUDA_ARCH` variable in
`config.cmake` to your desired cuda architecture to speed up the build.

To (optionally) build with MKL math library, please install MKL first based on
the guide in [Math Library Selection](build_from_source#math-library-selection).

**Step 3:** Build MXNet core shared library.

```bash
rm -rf build
mkdir -p build && cd build
cmake ..
cmake --build .
```

Specify `cmake --build . --parallel N` to set the number of parallel compilation
jobs. Default is derived from CPUs available.

After a successful build, you will find the `libmxnet.dylib` in the `build`
folder in your MXNet project root. `libmxnet.dylib` is required to install
language bindings described in the next section.

<hr>


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

<hr>


### Install MXNet for Python

To install the MXNet Python binding navigate to the root of the MXNet folder then run the following:

```bash
cd python
pip install --user -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means
that if you edit the source files, these changes will be reflected in the
package installed.


### Install the MXNet Package for C++

Refer to the [C++ Package setup guide](c_plus_plus).
<hr>


### Install the MXNet Package for Clojure

Refer to the [Clojure setup guide](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package).
<hr>


### Install the MXNet Package for R
Run the following commands to install the MXNet dependencies and build the MXNet
R package.

```r
    Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
```
```bash
    cd R-package
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    cd ..
    make -f R-package/Makefile rpkg
```

## Install the MXNet Package for Julia
The MXNet package for Julia is hosted in a separate repository, MXNet.jl, which is available on [GitHub](https://github.com/dmlc/MXNet.jl). To use Julia binding it with an existing libmxnet installation, set the ```MXNET_HOME``` environment variable by running the following command:

```bash
	export MXNET_HOME=/<path to>/libmxnet
```

The path to the existing libmxnet installation should be the root directory of libmxnet. In other words, you should be able to find the ```libmxnet.so``` file at ```$MXNET_HOME/lib```. For example, if the root directory of libmxnet is ```~```, you would run the following command:

```bash
	export MXNET_HOME=/~/libmxnet
```

You might want to add this command to your ```~/.bashrc``` file. If you do, you can install the Julia package in the Julia console using the following command:

```julia
	Pkg.add("MXNet")
```

For more details about installing and using MXNet with Julia, see the [MXNet Julia documentation]({{'/api/julia'|relative_url}}).


### Install the MXNet Package for Scala

To use the MXNet-Scala package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Scala Setup Instructions](scala_setup).

If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial]({{'/api/scala/docs/tutorials/mxnet_scala_on_intellij'|relative_url}}) instead.

### Install the MXNet Package for Perl

Before you build MXNet for Perl from source code, you must complete [building the shared library](#build-the-shared-library).
After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Perl package:

```bash
    brew install swig
    sudo sh -c 'curl -L https://cpanmin.us | perl - App::cpanminus'
    sudo cpanm -q -n PDL Mouse Function::Parameters Hash::Ordered PDL::CCS

    MXNET_HOME=${PWD}
    export PERL5LIB=${HOME}/perl5/lib/perl5

    cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make
    install_name_tool -change lib/libmxnet.so \
        ${MXNET_HOME}/lib/libmxnet.so \
        blib/arch/auto/AI/MXNetCAPI/MXNetCAPI.bundle
    make install

    cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make
    install_name_tool -change lib/libmxnet.so \
            ${MXNET_HOME}/lib/libmxnet.so \
            blib/arch/auto/AI/NNVMCAPI/NNVMCAPI.bundle
    make install

    cd ${MXNET_HOME}/perl-package/AI-MXNet/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install
```

## Contributions

You are more than welcome to contribute easy installation scripts for other operating systems and programming languages.
See the [community contributions page]({{'/community/contribute'|relative_url}}) for further information.

## Next Steps

* [Tutorials]({{'/api'|relative_url}})
* [How To]({{'/api/faq/add_op_in_backend'|relative_url}})
* [Architecture]({{'/api/architecture/overview'|relative_url}})
