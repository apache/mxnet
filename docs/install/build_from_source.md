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


## Build the shared library

### Prerequisites

You need C++ build tools and a BLAS library to build the MXNet shared library. If you want to run MXNet with GPUs, you will need to install [NVDIA CUDA and cuDNN](https://developer.nvidia.com/cuda-downloads) first.


#### C++ build tools

1. A C++ compiler that supports C++ 11.
[G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/) or
[Clang](http://clang.llvm.org/) is required.

2. [Git](https://git-scm.com/downloads) for downloading the sources from Github repository.

3. [GNU Make](https://www.gnu.org/software/make/) ([cmake](https://cmake.org/)
   for Windows) to build the library.


#### BLAS library

MXNet relies on the
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic
Linear Algebra Subprograms) library for numerical computations. You can install
any one among [ATLAS](http://math-atlas.sourceforge.net/),
[OpenBLAS](http://www.openblas.net/) and
[MKL](https://software.intel.com/en-us/intel-mkl).


#### Optional

* [OpenCV](http://opencv.org/) for Image Loading and Augmentation
* [NVDIA CUDA and cuDNN](https://developer.nvidia.com/cuda-downloads) for running MXNet with GPUs


### macOS

Refer to the [MXNet macOS setup guide](osx_setup.html) for detailed instructions.


### Windows

Refer to the [MXNet Windows setup guide](windows_setup.html) for detailed instructions.


### Ubuntu

Refer to the <a href="ubuntu_setup.html">MXNet Ubuntu installation guide</a> for build from source instructions as well as installation of language bindings.


### CentOS
1. Install build tools and git on `CentOS >= 7` and `Fedora >= 19`:

```bash
sudo yum groupinstall -y "Development Tools" && sudo yum install -y git
```

2. Install Atlas:

```bash
sudo yum install atlas-devel
```

### Other Linux
Installing both `git` and `make` by following instructions on the websites is
straightforward. Here we provide the instructions to build `gcc-4.8` from source codes.

1. Install the 32-bit `libc` with one of the following system-specific commands:

```bash
sudo apt-get install libc6-dev-i386 # In Ubuntu
sudo yum install glibc-devel.i686   # In RHEL (Red Hat Linux)
sudo yum install glibc-devel.i386   # In CentOS 5.8
sudo yum install glibc-devel.i686   # In CentOS 6/7
```

2. Download and extract the `gcc` source code with the prerequisites:

```bash
wget http://mirrors.concertpass.com/gcc/releases/gcc-4.8.5/gcc-4.8.5.tar.gz
tar -zxf gcc-4.8.5.tar.gz
cd gcc-4.8.5
./contrib/download_prerequisites
```

3. Build `gcc` by using 10 threads and then install to `/usr/local`

```bash
mkdir release && cd release
../configure --prefix=/usr/local --enable-languages=c,c++
make -j10
sudo make install
```

4. Add the lib path to your configure file such as `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
```

5. Build [OpenBLAS from source](https://github.com/xianyi/OpenBLAS#installation-from-source).

6. Build OpenCV

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


### Build

1. Clone the MXNet project.
```bash
git clone --recursive https://github.com/apache/incubator-mxnet mxnet
cd mxnet
```

There is a configuration file for make,
[`make/config.mk`](https://github.com/apache/incubator-mxnet/blob/master/make/config.mk), that contains all the compilation options. You can edit it and then run `make`.

To enable C++ package, just add `USE_CPP_PACKAGE=1` when you run `make`.

Other typical configurations are:

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

* Build on macOS with the default BLAS library and clang installed with `xcode` (OPENMP is disabled because it is not supported in default by clang):

```bash
make -j USE_BLAS=apple USE_OPENCV=0 USE_OPENMP=0
```


## Build MXNet using NCCL
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

### Validation
- Follow the steps to install MXNet Python binding.
- Comment the following line in ```test_nccl.py``` file at ```incubator-mxnet/tests/python/gpu/test_nccl.py```
``` bash
@unittest.skip("Test requires NCCL library installed and enabled during build")
```
- Run test_nccl.py script as follows. The test should complete. It does not produce any output.
``` bash
nosetests --verbose tests/python/gpu/test_nccl.py
```

### Recommendation for best performance
It is recommended to set environment variable NCCL_LAUNCH_MODE to PARALLEL when using NCCL version 2.1 or newer.


## Installing MXNet Language Bindings

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
