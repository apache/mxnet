# Build MXNet from Source

This document explains how to build MXNet from source code. Building MXNet from source is a two step process.

1. Build the MXNet shared library, `libmxnet.so`, from [C++ source files](#build-the-shared-library)
2. Install the [language bindings](#installing-mxnet-language-bindings) for MXNet. MXNet supports the following languages:
    - Python
    - C++
    - Clojure
    - Julia
    - Perl
    - R
    - Scala

## Prerequisites

You need C++ build tools and a BLAS library to build the MXNet shared library. If you want to run MXNet with GPUs, you will need to install [NVDIA CUDA and cuDNN](https://developer.nvidia.com/cuda-downloads) first.

You may use [GNU Make](https://www.gnu.org/software/make/) to build the library but [cmake](https://cmake.org/) is required when building with MKLDNN


### C++ build tools

1. A C++ compiler that supports C++ 11.
[G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/) or
[Clang](http://clang.llvm.org/) is required.

2. [Git](https://git-scm.com/downloads) for downloading the sources from Github repository.




### BLAS library

MXNet relies on the
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic
Linear Algebra Subprograms) library for numerical computations.
Those can be extended with [LAPACK (Linear Algebra Package)](https://github.com/Reference-LAPACK/lapack), an additional set of mathematical functions.

MXNet supports multiple mathematical backends for computations on the CPU:

* [Apple Accelerate](https://developer.apple.com/documentation/accelerate)
* [ATLAS](http://math-atlas.sourceforge.net/)
* [MKL](https://software.intel.com/en-us/intel-mkl) (MKL, MKLML)
* [MKL-DNN](https://github.com/intel/mkl-dnn)
* [OpenBLAS](http://www.openblas.net/)

Usage of these are covered in more detail in the [build configurations](#build-configurations) section.


### Optional

These might be optional, but they're typically desirable.

* [OpenCV](http://opencv.org/) for Image Loading and Augmentation
* [NVDIA CUDA and cuDNN](https://developer.nvidia.com/cuda-downloads) for running MXNet with GPUs


## Build Instructions by Operating System

Detailed instructions are provided per operating system.
You may jump to those, but it is recommended that you continue reading to understand more general build from source options.

| | | | |
|---|---|---|---|
| [macOS](osx_setup.html) | [Ubuntu](ubuntu_setup.html) | [CentOS/*unix](centos_setup.html) | [Windows](windows_setup.html) |
| [raspbian](raspian_setup.html) | [tx2](tx2_setup.html) | | |



## Build

1. Clone the MXNet project.
```bash
git clone --recursive https://github.com/apache/incubator-mxnet mxnet
cd mxnet
```

There is a configuration file for make,
[`make/config.mk`](https://github.com/apache/incubator-mxnet/blob/master/make/config.mk), that contains all the compilation options. You can edit it and then run `make`.


## Build Configurations

`cmake` is recommended for building MXNet (and is required to build with MKLDNN), however you may use `make` instead.


### Math Library Selection
It is useful to consider your math library selection first.

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

Intel's MKL (Math Kernel Library) is one of the most powerful math libraries
https://software.intel.com/en-us/mkl

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
Register and download on the [Intel performance libraries website](https://software.seek.intel.com/performance-libraries).

Note: MKL is supported only for desktop builds and the framework itself supports the following
hardware:

* Intel® Xeon Phi™ processor
* Intel® Xeon® processor
* Intel® Core™ processor family
* Intel Atom® processor

If you have a different processor you can still try to use MKL, but performance results are
unpredictable.


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
echo "USE_NCCP_PATH=path-to-nccl-installation-folder" >> make/config.mk
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


### Build MXNet with Language Packages
* To enable C++ package, just add `USE_CPP_PACKAGE=1` when you run `make` or `cmake`.


### Usage Examples
* `-j` runs multiple jobs against multi-core CPUs. Example using all cores on Linux:

```bash
make -j$(nproc)
```

* Build without using OpenCV:

```bash
make USE_OPENCV=0
```

* Build with both OpenBLAS, GPU, and OpenCV support:

```bash
make -j USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
```

* Build on **macOS** with the default BLAS library (Apple Accelerate) and Clang installed with `xcode` (OPENMP is disabled because it is not supported by the Apple version of Clang):

```bash
make -j USE_BLAS=apple USE_OPENCV=0 USE_OPENMP=0
```

* To use OpenMP on **macOS** you need to install the Clang compiler, `llvm` (the one provided by Apple does not support OpenMP):

```bash
brew install llvm
make -j USE_BLAS=apple USE_OPENMP=1
```

## Installing MXNet Language Bindings
After building MXNet's shared library, you can install other language bindings. (Except for C++. You need to build this when you build MXNet from source.)

The following table provides links to each language binding by operating system:
|   | Linux | macOS | Windows |
|---|---|---|---|
| Python | [Linux](ubuntu_setup.html#install-mxnet-for-python) | [macOS](osx_setup.html) | [Windows](windows_setup.html#install-mxnet-for-python) |
| C++ | [Linux](c_plus_plus.html) | [macOS](c_plus_plus.html) | [Windows](c_plus_plus.html) |
| Clojure | [Linux](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package) | [macOS](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package) | n/a |
| Julia | [Linux](ubuntu_setup.html#install-the-mxnet-package-for-julia) | [macOS](osx_setup.html#install-the-mxnet-package-for-julia) | [Windows](windows_setup.html#install-the-mxnet-package-for-julia) |
| Perl | [Linux](ubuntu_setup.html#install-the-mxnet-package-for-perl) | [macOS](osx_setup.html#install-the-mxnet-package-for-perl) | [Windows](n/a) |
| R | [Linux](ubuntu_setup.html#install-the-mxnet-package-for-r) | [macOS](osx_setup.html#install-the-mxnet-package-for-r) | [Windows](windows_setup.html#install-the-mxnet-package-for-r) |
| Scala | [Linux](scala_setup.html) | [macOS](scala_setup.html) | n/a |
