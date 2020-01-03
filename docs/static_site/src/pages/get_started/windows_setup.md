---
layout: page
title: Windows Setup
action: Get Started
action_url: /get_started
permalink: /get_started/windows_setup
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

# Installing MXNet on Windows

The following describes how to install with pip for computers with CPUs, Intel CPUs, and NVIDIA GPUs. Further along in the document you can learn how to build MXNet from source on Windows, or how to install packages that support different language APIs to MXNet.

- [Prerequisites](#prerequisites)
- [Install MXNet with Python](#install-mxnet-with-python)
    - [Install with CPUs](#install-with-cpus)
    - [Install with Intel CPUs](#install-with-intel-cpus)
    - [Install with NVIDIA GPUs](#install-with-nvidia-gpus)
    - [Notes on the Python Packages](#notes-on-the-python-packages)
- [Build from Source](#build-from-source)
- Install MXNet with a Programming Language API
    - [Python](#install-the-mxnet-package-for-python)
    - [R](#install-the-mxnet-package-for-r)
    - [Julia](#install-the-mxnet-package-for-julia)


## Prerequisites

### Minimum System Requirements

* Windows 7<sup><a href="#fn1" id="ref1">1</a></sup>, 10, Server 2012 R2, or Server 2016
* Visual Studio 2015 or 2017 (any type)
* Python 2.7 or 3.6
* pip

<sup id="fn1">1. There are [known issues](https://github.com/apache/incubator-mxnet/issues?utf8=%E2%9C%93&q=is%3Aissue+windows7+label%3AWindows+) with Windows 7. <a href="#ref1" title="Return to source text.">↩</a></sup>

### Recommended System Requirements

* Windows 10, Server 2012 R2, or Server 2016
* Visual Studio 2017 (any type)
* At least one [NVIDIA CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus)
* MKL-enabled CPU: Intel® Xeon® processor, Intel® Core™ processor family, Intel Atom® processor, or Intel® Xeon Phi™ processor
* Python 2.7 or 3.6
* pip


## Install MXNet with Python

The easiest way to install MXNet on Windows is by using a [Python pip package](https://pip.pypa.io/en/stable/installing/).

**Note**: Windows pip packages typically release a few days after a new version MXNet is released. Make sure you verify which version gets installed.

### Install with CPUs

Install MXNet with CPU support with Python:

```bash
pip install mxnet
```

Now [validate your MXNet installation with Python](/get_started/validate_mxnet).

### Install with Intel CPUs

MXNet has experimental support for Intel [MKL](https://software.intel.com/en-us/mkl) and [MKL-DNN](https://github.com/intel/mkl-dnn). When using supported Intel hardware, inference and training can be vastly faster when using MXNet with [MKL](https://software.intel.com/en-us/mkl) or [MKL-DNN](https://github.com/intel/mkl-dnn).

The following steps will setup MXNet with MKL. MKL-DNN can be enabled only when building from source.
1. Download and install [Intel MKL](https://software.intel.com/en-us/mkl/choose-download/windows) (registration required).
1. Install MXNet with MKL support with Python:

```bash
pip install mxnet-mkl
```

Now [validate your MXNet installation with Python](/get_started/validate_mxnet).

### Install with NVIDIA GPUs

When using supported NVIDIA GPU hardware, inference and training can be vastly faster with [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn). You have two options for installing MXNet with CUDA support with a Python package.
- [Install with CUDA support](#install-with-cuda-support)
- [Install with CUDA and MKL support](#install-with-cuda-and-mkl-support)

#### Install with CUDA Support

The following steps will setup MXNet with CUDA. cuDNN can be enabled only when building from source.
1. Install [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) or [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/).
1. Download and install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal). CUDA versions 9.2 or 9.0 are recommended. Some [issues with CUDA 9.1](https://github.com/apache/incubator-mxnet/labels/CUDA) have been identified in the past.
1. Download and install [NVIDIA_CUDA_DNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)
1. Install MXNet with CUDA support with pip:

```bash
pip install mxnet-cu92
```

Once you have installed a version of MXNet, [validate your MXNet installation with Python](validate_mxnet).

#### Install with CUDA and MKL Support

You can also use a combination of CPU/GPU enhancements provided by Intel and NVIDIA.

The following steps will setup MXNet with CUDA and MKL.
1. Install [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) or [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/).
1. Download and install [Intel MKL](https://software.intel.com/en-us/mkl/choose-download/windows) (registration required).
1. Download and install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal).
1. Download and install [NVIDIA_CUDA_DNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)
1. Install MXNet with MKL support with pip:

```bash
pip install mxnet-cu92mkl
```

Once you have installed a version of MXNet, [validate your MXNet installation with Python](validate_mxnet).

### Notes on the Python Packages
To get further enhancements for deep neural networks, you may want to enable MKL-DNN and/or cuDNN. Each of these require you to [build from source](#build-from-source) and to enable the build flags for each.

Check the chart below for other options or refer to [PyPI for other MXNet pip packages](https://pypi.org/project/mxnet/).

![pip packages](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/install/pip-packages.png)


## Build from Source


For automated setting up of developer environment in windows, use script bundle from the
![ci/windows_dev_env](https://github.com/apache/incubator-mxnet/tree/master/ci/windows_dev_env/)
folder. Copy to a local directory and execute:

```
.\setup.ps1
```

This will install the recommended VS Community, Python, git, and other dependencies needed to build in windows.
After that, follow the steps below starting from "build the MXNet source code" section below.

```
C:\Python37\python.exe .\ci\build_windows.py
```

These commands produce a library called ```mxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.


We have installed MXNet core library. Next, we will install MXNet interface package for programming language of your choice:
- [Python](#install-the-mxnet-package-for-python)
- [R](#install-the-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- **Scala** is not yet available for Windows

### Optional step:

install ```graphviz``` library that we use for visualizing network graphs you build on MXNet. We will also install [Jupyter Notebook](http://jupyter.readthedocs.io/)  used for running MXNet tutorials and examples.
- Install ```graphviz``` by downloading MSI installer from [Graphviz Download Page](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).
**Note** Make sure to add graphviz executable path to PATH environment variable. Refer [here for more details](http://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)
- Install ```Jupyter``` by installing [Anaconda for Python 2.7](https://www.anaconda.com/download/)
**Note** Do not install Anaconda for Python 3.5. MXNet has a few compatibility issues with Python 3.5.


## Install the MXNet Package for Python

Use setup.py to install the package.
```bash
    # Assuming you are in root mxnet source code folder
    pip install --upgrade --force-reinstall -e python
```

Done! We have installed MXNet with Python interface.

You can continue with using MXNet-Python, or if you want to try a different language API for MXNet, keep reading.

## Install the MXNet Package for R
MXNet for R is available for both CPUs and GPUs.

### Installing MXNet-R on a Computer with a CPU Processor

To install MXNet on a computer with a CPU processor, choose from two options:

* Use the prebuilt binary package
* Build the library from source code

#### Installing MXNet-R with the Prebuilt Binary Package(CPU)
For Windows users, MXNet provides prebuilt binary packages.
You can install the package directly in the R console.

Note: packages for 3.6.x are not yet available.
Install 3.5.x of R from [CRAN](https://cran.r-project.org/bin/windows/base/old/).

For CPU-only package:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
  options(repos = cran)
  install.packages("mxnet")
```

#### Building MXNet-R from Source Code(CPU)
1. Clone the MXNet github repo.
```sh
git clone --recursive https://github.com/apache/incubator-mxnet
```
The `--recursive` is to clone all the submodules used by MXNet. You will be editing the ```"/mxnet/R-package"``` folder.
2. Download prebuilt GPU-enabled MXNet libraries for Windows from [Windows release](https://github.com/yajiedesign/mxnet/releases). You will need `mxnet_x64_vc14_cpu.7z` and `prebuildbase_win10_x64_vc14.7z` where X stands for your CUDA toolkit version
3. Create a folder called ```R-package/inst/libs/x64```. MXNet supports only 64-bit operating systems, so you need the x64 folder.
4. Copy the following shared libraries (.dll files) into the ```R-package/inst/libs/x64``` folder.
```
libgcc_s_seh-1.dll
libgfortran-3.dll
libmxnet.dll
libmxnet.lib
libopenblas.dll
libquadmath-0.dll
mxnet.dll
unzip.exe
unzip32.dll
vcomp140.dll
wget.exe
```
These dlls can be found in `prebuildbase_win10_x64_vc14/3rdparty`, `mxnet_x64_vc14_cpu/build`, `mxnet_x64_vc14_cpu/lib`.
5. Copy the header files from `dmlc`, `mxnet`, `mxshadow` and `nnvm` from `mxnet_x64_vc14_cpu/include and mxnet_x64_vc14_cpu/nvnm/include` into `./R-package/inst/include`. It should look like
```
./R-package/inst
└── include
    ├── dmlc
    ├── mxnet
    ├── mshadow
    └── nnvm
```

6. Make sure that R executable is added to your ```PATH``` in the environment variables. Running the ```where R``` command at the command prompt should return the location.
7. Also make sure that Rtools is installed and the executable is added to your ```PATH``` in the environment variables.
8. Temporary patch - im2rec currently results in crashes during the build. Remove the im2rec.h and im2rec.cc files in R-package/src/ from cloned repository and comment out the two im2rec lines in [R-package/src/mxnet.cc](https://github.com/apache/incubator-mxnet/blob/master/R-package/src/mxnet.cc) as shown below.
```
#include "./kvstore.h"
#include "./export.h"
//#include "./im2rec.h"
......
......
  DataIterCreateFunction::InitRcppModule();
  KVStore::InitRcppModule();
  Exporter::InitRcppModule();
//IM2REC::InitRcppModule();
}
```

9. Now open the Windows CMD with admin rights and change the directory to the `mxnet` folder(cloned repository). Then use the following commands to build R package:

```bash
echo import(Rcpp) > R-package\NAMESPACE
echo import(methods) >> R-package\NAMESPACE
Rscript -e "install.packages('devtools', repos = 'https://cloud.r-project.org')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cloud.r-project.org')); install_deps(dependencies = TRUE)"
cd ..

R CMD INSTALL --no-multiarch R-package

Rscript -e "require(mxnet); mxnet:::mxnet.export('R-package')"
rm R-package/NAMESPACE
Rscript -e "require(devtools); install_version('roxygen2', version = '5.0.1', repos = 'https://cloud.r-project.org/', quiet = TRUE)"
Rscript -e "require(roxygen2); roxygen2::roxygenise('R-package')"

R CMD INSTALL --build --no-multiarch R-package
```

### Installing MXNet-R on a Computer with a GPU Processor
To install MXNet on a computer with a GPU processor, choose from two options:

* Use the prebuilt binary package
* Build the library from source code

However, a few dependencies remain for both options.  You will need the following:
* Install [Nvidia-drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us) if not installed. Latest driver based on your system configuration is recommended.

* Install [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/) (VS2015 or VS2017 is required by CUDA)

* Install  [NVidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)(cu92 is recommended though we support cu80, cu90, cu91 and cu92)

* Download and install [CuDNN](https://developer.nvidia.com/cudnn) (to provide a Deep Neural Network library). Latest version recommended.

Note: A pre-requisite to above softwares is [Nvidia-drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us) which we assume is installed.

#### Installing MXNet-R with the Prebuilt Binary Package(GPU)
For Windows users, MXNet provides prebuilt binary packages.
You can install the package directly in the R console after you have the above software installed.

Note: packages for 3.6.x are not yet available.
Install 3.5.x of R from [CRAN](https://cran.r-project.org/bin/windows/base/old/).

For GPU package:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
  options(repos = cran)
  install.packages("mxnet")
```
Change cu92 to cu80, cu90 or cu91 based on your CUDA toolkit version. Currently, MXNet supports these versions of CUDA.

#### Building MXNet-R from Source Code(GPU)
After you have installed above software, continue with the following steps to build MXNet-R:
1. Clone the MXNet github repo.
```sh
git clone --recursive https://github.com/apache/incubator-mxnet
```
The `--recursive` is to clone all the submodules used by MXNet. You will be editing the ```"/mxnet/R-package"``` folder.
2. Download prebuilt GPU-enabled MXNet libraries for Windows from https://github.com/yajiedesign/mxnet/releases. You will need `mxnet_x64_vc14_gpu_cuX.7z` and `prebuildbase_win10_x64_vc14.7z` where X stands for your CUDA toolkit version
3. Create a folder called ```R-package/inst/libs/x64```. MXNet supports only 64-bit operating systems, so you need the x64 folder.
4. Copy the following shared libraries (.dll files) into the ```R-package/inst/libs/x64``` folder:
```
libgcc_s_seh-1.dll
libgfortran-3.dll
libmxnet.dll
libmxnet.lib
libopenblas.dll
libquadmath-0.dll
mxnet.dll
unzip.exe
unzip32.dll
vcomp140.dll
wget.exe
```
These dlls can be found in `prebuildbase_win10_x64_vc14/3rdparty`, `mxnet_x64_vc14_gpu_cuX/build`, `mxnet_x64_vc14_gpu_cuX/lib`.
5. Copy the header files from `dmlc`, `mxnet`, `mxshadow` and `nnvm` from `mxnet_x64_vc14_gpuX/include` and `mxnet_x64_vc14_gpuX/nvnm/include` into `./R-package/inst/include`. It should look like:
```
./R-package/inst
└── include
    ├── dmlc
    ├── mxnet
    ├── mshadow
    └── nnvm
```
6. Make sure that R executable is added to your ```PATH``` in the environment variables. Running the ```where R``` command at the command prompt should return the location.
7. Also make sure that Rtools is installed and the executable is added to your ```PATH``` in the environment variables.
8. Temporary patch - im2rec currently results in crashes during the build. Remove the im2rec.h and im2rec.cc files in R-package/src/ from cloned repository and comment out the two im2rec lines in [R-package/src/mxnet.cc](https://github.com/apache/incubator-mxnet/blob/master/R-package/src/mxnet.cc) as shown below.
```bash
#include "./kvstore.h"
#include "./export.h"
//#include "./im2rec.h"
......
......
  DataIterCreateFunction::InitRcppModule();
  KVStore::InitRcppModule();
  Exporter::InitRcppModule();
//IM2REC::InitRcppModule();
}
```
9. Now open the Windows CMD with admin rights and change the directory to the `mxnet` folder(cloned repository). Then use the following commands
to build R package:

```bat
echo import(Rcpp) > R-package\NAMESPACE
echo import(methods) >> R-package\NAMESPACE
Rscript -e "install.packages('devtools', repos = 'https://cloud.r-project.org')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cloud.r-project.org')); install_deps(dependencies = TRUE)"
cd ..

R CMD INSTALL --no-multiarch R-package

Rscript -e "require(mxnet); mxnet:::mxnet.export('R-package')"
rm R-package/NAMESPACE
Rscript -e "require(devtools); install_version('roxygen2', version = '5.0.1', repos = 'https://cloud.r-project.org/', quiet = TRUE)"
Rscript -e "require(roxygen2); roxygen2::roxygenise('R-package')"

R CMD INSTALL --build --no-multiarch R-package
```

**Note:** To maximize its portability, the MXNet library is built with the Rcpp end. Computers running Windows need [MSVC](https://en.wikipedia.org/wiki/Visual_C%2B%2B) (Microsoft Visual C++) to handle CUDA toolchain compatibilities.

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


## Installing the MXNet Package for Scala

MXNet-Scala is not yet available for Windows.
