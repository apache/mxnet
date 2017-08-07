# Build MXNet from Source

**NOTE:** For MXNet with Python installation, please refer to the [new install guide](http://mxnet.io/get_started/install.html).

This document explains how to build MXNet from sources. Building MXNet from sources is a 2 step process.

1. Build the MXNet shared library, `libmxnet.so`, from [C++ source files](#build-the-shared-library)
2. Install the language binding for MXNet. MXNet supports -
   [C++](#build-the-c-package),
   [Scala](#build-the-scala-package), [R](#build-the-r-package), and
   [Julia](#build-the-julia-package).

## Build the shared library

### Prerequisites

You need C++ build tools and BLAS library to build MXNet shared library. If you want to run MXNet on GPUs, you need to install CUDA and CuDNN.

#### C++ build tools

1. A C++ compiler that supports C++ 11.
[G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/) or
[Clang](http://clang.llvm.org/) is required.

2. [Git](https://git-scm.com/downloads) for downloading the sources from Github repository.

3. [GNU Make](https://www.gnu.org/software/make/) ([cmake](https://cmake.org/)
   for Windows) to build the library.


Select your preferences and follow the instructions to install MXNet from sources.
<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt active">Linux</button>
<button type="button" class="btn btn-default opt">macOS</button>
<button type="button" class="btn btn-default opt">Windows</button>
</div>
<script type="text/javascript" src='../../_static/js/options.js'></script>

<div class="linux">

Then select the Linux distribution:
<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt active">Ubuntu</button>
<button type="button" class="btn btn-default opt">CentOS</button>
<button type="button" class="btn btn-default opt">Others</button>
</div>

- **Ubuntu** for systems supporting the `apt-get`
  package management program
- **CentOS** for systems supporting the `yum` package
  management program
- **Others** for general Linux-like systems building dependencies from scratch.

<div class="ubuntu">

Install build tools and git on `Ubuntu >= 13.10` and `Debian >= 8`.

```bash
sudo apt-get update && sudo apt-get install build-essential git
```

</div>

<div class="centos">

Install build tools and git on `CentOS >= 7` and `Fedora >= 19`.

```bash
sudo yum groupinstall -y "Development Tools" && sudo yum install -y git
```

</div>

<div class="others">

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

</div>
</div> <!-- linux -->

<div class="windows">

1. If [Microsoft Visual Studio 2015](https://www.visualstudio.com/downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and Install [CMake](https://cmake.org/) if it is not already installed.

</div>

<div class="macos">

Install [Xcode](https://developer.apple.com/xcode/).

</div>

#### BLAS library

MXNet relies on the
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic
Linear Algebra Subprograms) library for numerical computations. You can install
any one among [ATLAS](http://math-atlas.sourceforge.net/),
[OpenBLAS](http://www.openblas.net/) and
[MKL](https://software.intel.com/en-us/intel-mkl).

<div class="linux">
<div class="ubuntu">

```bash
sudo apt-get install libatlas-base-dev
```

</div>

<div class="centos">

```bash
sudo yum install atlas-devel
```

</div>

<div class="linux">

You can follow this link to build
[OpenBlas from source](https://github.com/xianyi/OpenBLAS#installation-from-source).

</div>
</div>

<div class="macos">

macOS users can skip this step as `xcode` ships with a BLAS library.

</div>

<div class="windows">

1. Download pre-built binaries for [OpenBLAS](https://sourceforge.net/projects/openblas/files/)
2. Set the environment variable `OpenBLAS_HOME` to point to the OpenBLAS
   directory that contains the `include/` and `lib/` directories. Typically, you
   can find the directory in `C:\Program files (x86)\OpenBLAS\`.

</div>

#### Optional: [OpenCV](http://opencv.org/) for Image Loading and Augmentation

<div class="linux">
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

<div class="others">

To build OpenCV from source code, you need the [cmake](https://cmake.org) library.

1. If you don't have cmake or if your version of cmake is earlier than 3.6.1, run the following commands to install a newer version of cmake:

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
</div>

<div class="windows">

First download and install [OpenCV](http://opencv.org/releases.html), then set
the environment variable `OpenCV_DIR` to point to the OpenCV build directory.

</div>

#### Optional: [CUDA](https://developer.nvidia.com/cuda-downloads)/[cuDNN](https://developer.nvidia.com/cudnn) for Nvidia GPUs

MXNet is compatible with both CUDA 7.5 and 8.0. It is recommended to use cuDNN 5.

<div class="linux">
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
</div>

### Build

<div class="linux macos">

First clone the recent codes

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet
```

File
[`make/config.mk`](https://github.com/dmlc/mxnet/blob/master/make/config.mk)
contains all the compilation options. You can edit it and then `make`. There are
some example build options

If you want to build MXNet with C++ language binding, please make sure you read [Build the C++ package](#build-the-c-package) first.

</div>

<div class="linux">

- Build without using OpenCV. `-j` runs multiple jobs against multi-core CPUs.

  ```bash
  make -j USE_OPENCV=0
  ```

- Build with both GPU and OpenCV support

  ```bash
  make -j USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
  ```

</div>

<div class="macos">

- Build with the default BLAS library and clang installed with `xcode` (OPENMP
  is disabled because it is not supported in default by clang).

  ```bash
  make -j USE_BLAS=apple USE_OPENCV=0 USE_OPENMP=0
  ```

</div>

<div class="windows">

Use [CMake](https://cmake.org/) to create a Visual Studio solution in ```./build```.

In Visual Studio, open the solution file,```.sln```, and compile it.
These commands produce a library called ```mxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.

</div>

## Build the C++ package
The C++ package has the same prerequisites as the MXNet library, you should also have `python` installed. (Both `python` 2 and 3 are supported)

To enable C++ package, just add `USE_CPP_PACKAGE=1` in the build options when building the MXNet shared library.

## Build the R package

The R package requires `R` to be installed.

<div class="ubuntu">

Follow the below instructions to install the latest R on Ubuntu 14.04 (trusty) and also the libraries used
to build other R package dependencies.

```bash
echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | apt-key add -

apt-get update
apt-get install -y r-base r-base-dev libxml2-dev libxt-dev libssl-dev
```

</div>

Install the required R package dependencies:

```bash
cd R-package
Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
```

Next, build and install the MXNet R package:

```bash
cd ..
make rpkg
R CMD INSTALL mxnet_current_r.tar.gz
```

## Build the Scala package

Both JDK and Maven are required to build the Scala package.

<div class="ubuntu">

```bash
sudo apt-get install -y maven default-jdk
```

</div>

The following command builds the `.jar` package:

```bash
make scalapkg
```

which can be found by `ls scala-package/assembly/*/target/*SNAPSHOT.jar`.

Optionally, we can install Scala for the interactive interface.

<div class="ubuntu">

```bash
wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb
dpkg -i scala-2.11.8.deb
rm scala-2.11.8.deb
```

</div>

Then we can start `scala` with `mxnet` imported by

```bash
scala -cp scala-package/assembly/*/target/*SNAPSHOT.jar
```

## Build the Julia package

We need to first install Julia.

<div class="ubuntu centos linux">

The following commands install Julia 0.5.1

```bash
wget -q https://julialang.s3.amazonaws.com/bin/linux/x64/0.5/julia-0.5.1-linux-x86_64.tar.gz
tar -zxf julia-0.5.1-linux-x86_64.tar.gz
rm julia-0.5.1-linux-x86_64.tar.gz
ln -s $(pwd)/julia-6445c82d00/bin/julia /usr/bin/julia
```

</div>

Next set the environment variable `MXNET_HOME=/path/to/mxnet` so that Julia
can find the pre-built library.

Install the Julia package for MXNet with:

```bash
julia -e 'Pkg.add("MXNet")'
```

### Build the Perl package

Run the following command from the MXNet source root directory to build the MXNet Perl package:

```bash
    sudo apt-get install libmouse-perl pdl cpanminus swig libgraphviz-perl
    cpanm -q -L "${HOME}/perl5" Function::Parameters

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
