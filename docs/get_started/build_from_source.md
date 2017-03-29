# Build MXNet from Source

This document explains how to build MXNet from source codes for the following
operation systems. Please select the one you like.

<div class='text-center'>
<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt active">Ubuntu</button>
<button type="button" class="btn btn-default opt">CentOS</button>
<button type="button" class="btn btn-default opt">Linux</button>
<button type="button" class="btn btn-default opt">macOS</button>
<button type="button" class="btn btn-default opt">Windows</button>
</div>
</div>

- **Ubuntu** for systems such as Ubuntu and Debian that use the `apt-get`
  package management program
- **CentOS** for systems such as Centos, RHEL and Fedora that use the `yum` package
  management program
- **Linux** for general Linux-like systems in which we build all dependencies
  from source codes.
- **macOS** for Mac operating system (named Mac OS X before)
- **Windows** for Microsoft Windows

The whole process mainly contains two steps:

1. Build the shared `libmxnet` library from C++ source files
2. Build the front-end language package such as Python, Scala, R and Julia.

## Build the shared libray

### Prerequisites

The minimum requirements to build MXNet's shared library include a C++ compiler
and a BLAS library. There are optional dependencies for enhanced features.

#### C++ build tools

A C++ compiler that supports C++ 11 such as
[G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/) and
[Clang](http://clang.llvm.org/) is required.

<div class="ubuntu">

For `Ubuntu >= 13.10` and `Debian >= 8` you can install it by

```bash
sudo apt-get update && sudo apt-get install build-essential
```

Refer to `Linux` to build `gcc` from source codes for lower version systems.

</div>

<div class="centos">

For `CentOS >= 7` and `Fedora >= 19`, you can install it by

```bash
sudo yum groupinstall -y "Development Tools"
```

Refer to `Linux` to build `gcc` from source codes for lower version systems.

</div>


<div class="linux">

The following instructions build `gcc-4.8` from source codes.

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

</div>

<div class="windows">

1. If [Microsoft Visual Studio 2013](https://www.visualstudio.com/downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Install [Visual C++ Compiler Nov 2013 CTP](https://www.microsoft.com/en-us/download/details.aspx?id=41151).
3. Back up all of the files in the ```C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC``` folder to a different location.
4. Copy all of the files in the ```C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP``` folder (or the folder where you extracted the zip archive) to the ```C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC``` folder, and overwrite all existing files.

</div>

#### BLAS library

A [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic
Linear Algebra Subprograms) library is also required. Common choices include
[ATLAS](http://math-atlas.sourceforge.net/),
[OpenBLAS](http://www.openblas.net/) and
[MKL](https://software.intel.com/en-us/intel-mkl). Installing any one of them is
good enough.

<div class="ubuntu">

```bash
sudo apt-get install libopenblas-dev    # for openblas
sudo apt-get install libatlas-base-dev  # for atlas
```

</div>

<div class="centos">

```bash
sudo yum install atlas-devel  # for atlas
```

Installing OpenBLAS needs additional steps. First find the system version. For example,
if `cat /etc/*release* | grep VERSION` returns `7`, then

```bash
wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo rpm -Uvh epel-release-latest-7.noarch.rpm
sudo yum install -y openblas-devel
```

</div>

<div class="linux">

One can follow this link to build
[OpenBlas from source](https://github.com/xianyi/OpenBLAS#installation-from-source).

</div>

<div class="macos">

`xcode` ships with a BLAS library, macOS users can skip this step.

</div>

<div class="windows">

1. Download pre-build binaries for [OpenBLAS](https://sourceforge.net/projects/openblas/files/)
2. Set the environment variable `OpenBLAS_HOME` to point to the OpenBLAS
   directory that contains the `include/` and `lib/` directories. Typically, you
   can find the directory in `C:\Program files (x86)\OpenBLAS\`.

</div>

#### git

One can use `git` to download the source codes.

<div class="ubuntu">

```bash
sudo apt-get install git
```

</div>

<div class="centos">

```bash
sudo yum install git
```

</div>

<div class="linux macos windows">

Git can be downloaded on [git-scm.com](https://git-scm.com/downloads).

</div>

#### Optional: [OpenCV](http://opencv.org/) for Image Loading and Augmentation

<div class="ubuntu">

```bash
sudo apt-get install libopencv-dev
```

</div>


<div class="centos">

```bash
sudo apt-get install opencv-devel
```

</div>


<div class="linux">

To build OpenCV from source code, you need the [cmake](https://cmake.org) library .

1. (optional) If you don't have cmake or if your version of cmake is earlier than 3.6.1, run the following commands to install a newer version of cmake:

   ```bash
   wget https://cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.tar.gz
   tar -zxvf cmake-3.6.1-Linux-x86_64.tar.gz
   alias cmake="cmake-3.6.1-Linux-x86_64/bin/cmake"
   ```

2. To download and extract the OpenCV source code, run the following commands:

   ```bash
   wget https://codeload.github.com/opencv/opencv/zip/2.4.13
   unzip 2.4.13
   cd opencv-2.4.13
   mkdir release
   cd release/
   ```

3. Build OpenCV. The following commands build OpenCV with 10 threads. We
   disabled GPU support, which might significantly slow down an MXNet program
   running on a GPU processor. It also disables 1394 which might generate a
   warning. Then install it on `/usr/local`.

   ```bash
   cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
   make -j10
   sudo make install
   ```

4. Add the lib path to your configuration such as `~/.bashrc`.

   ```bash
   export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
   ```

</div>

<div class="windows">

First download and install [OpenCV](http://opencv.org/releases.html), then set the environment variable `OpenCV_DIR` to point to the OpenCV build directory.

</div>

#### Optional: [CUDA](https://developer.nvidia.com/cuda-downloads)/[cuDNN](https://developer.nvidia.com/cudnn) for Nvidia GPUs

<div class="ubuntu">

Install CUDA 7.5 and cuDNN 5 on Ubuntu 14.04

```bash
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" | sudo tee /etc/apt/sources.list.d/nvidia-ml.list
sudo apt-get update
sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`
sudo apt-get install -y cuda libcudnn5-dev=5.0.5-1+cuda7.5
```

</div>

### Build

<div class="ubuntu centos linux macos">

First clone the recent codes

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet
```

File
[`make/config.mk`](https://github.com/dmlc/mxnet/blob/master/make/config.mk)
contains all compilation options. You can edit it and then `make`. There are
some example build options

- Build with 10 threads, OpenBlas, without OpenCV

  ```bash
  make -j10 USE_BLAS=openblas USE_OPENCV=0
  ```

- Build

  ```bash
  make -j10 USE_BLAS=apple UES_OPENCV=0 USE_OPENMP=0
  ```

- Build

  ```bash
  make -j$(nproc) USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
  ```

</div>

<div class="windows">

Use [CMake](https://cmake.org/) to create a Visual Studio solution in ```./build```.

In Visual Studio, open the solution file,```.sln```, and compile it.
These commands produce a library called ```mxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.

</div>

## Build the Python package

The Python package requires both `python` and `numpy` are installed.

## Build the R package

## Build the Scala package

## Build the Julia package

<script type="text/javascript" src='../../_static/js/options.js'></script>
