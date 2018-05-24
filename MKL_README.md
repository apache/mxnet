* [1. Build/Install MXNet with MKLDNN on Linux](#1)
* [2. Build/Install MXNet with MKLDNN on MacOS](#2)
* [3. Build/Install MXNet with MKLDNN on Windows](#3)
* [4. Verify MXNet with python](#4)
* [5. Build/Install MXNet with a full MKL installation](#5)

<h2 id="1">Build/Install MXNet with MKLDNN on Linux</h2>

### Prerequisites

```
apt-get update && apt-get install -y build-essential git libopencv-dev curl gcc libopenblas-dev python python-pip python-dev python-opencv graphviz python-scipy python-sklearn
```

### Clone MXNet sources

```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git submodule update --recursive --init
```

### Build MXNet with MKLDNN

```
make -j $(nproc) USE_OPENCV=1 USE_MKLDNN=1 USE_BLAS=openblas USE_PROFILER=1
```

If you want to use Intel MKL blas, you can set `USE_BLAS=mkl USE_INTEL_PATH=/opt/intel`.

<h2 id="2">Build/Install MXNet with MKLDNN on MacOS</h2>

### Prerequisites

Install the dependencies, required for MXNet, with the following commands:

- Homebrew
- gcc (clang in macOS does not support openMP)
- OpenCV (for computer vision operations)

```
# Paste this command in Mac terminal to install Homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# install dependency
brew update
brew install pkg-config
brew install graphviz
brew tap homebrew/core
brew install opencv
brew tap homebrew/versions
brew install gcc49
brew link gcc49
```

### Enable openMP for MacOS

If you want to enable openMP for better performance, you should modify these two files:

1. Makefile L138:

```
ifeq ($(USE_OPENMP), 1)
	# ifneq ($(UNAME_S), Darwin)
		CFLAGS += -fopenmp
	# endif
endif
```

2. prepare_mkldnn.sh L96:

```
CC=gcc-4.9 CXX=g++-4.9 cmake $MKLDNN_ROOTDIR -DCMAKE_INSTALL_PREFIX=$MKLDNN_INSTALLDIR -B$MKLDNN_BUILDDIR -DARCH_OPT_FLAGS="-mtune=generic" -DWITH_TEST=OFF -DWITH_EXAMPLE=OFF >&2
```

### Build MXNet with MKLDNN

```
make -j $(sysctl -n hw.ncpu) USE_OPENCV=0 USE_OPENMP=1 USE_MKLDNN=1 USE_BLAS=apple USE_PROFILER=1
```

*Note: Temporarily disable OPENCV.*

<h2 id="3">Build/Install MXNet with MKLDNN on Windows</h2>

To build and install MXNet yourself, you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and Install [CMake](https://cmake.org/) if it is not already installed.
3. Download and install [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download).
4. Unzip the OpenCV package.
5. Set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (```C:\opencv\build\x64\vc14``` for example). Also, you need to add the OpenCV bin directory (```C:\opencv\build\x64\vc14\bin``` for example) to the ``PATH`` variable.
6. If you have Intel Math Kernel Library (MKL) installed, set ```MKL_ROOT``` to point to ```MKL``` directory that contains the ```include``` and ```lib```. If you want to use MKL blas, you should set ```-DUSE_BLAS=mkl``` when cmake. Typically, you can find the directory in
```C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\mkl```.
7. If you don't have the Intel Math Kernel Library (MKL) installed, download and install [OpenBLAS](http://sourceforge.net/projects/openblas/files/v0.2.14/). Note that you should also download ```mingw64.dll.zip`` along with openBLAS and add them to PATH.
8. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories. Typically, you can find the directory in ```C:\Program files (x86)\OpenBLAS\```. 

After you have installed all of the required dependencies, build the MXNet source code:

1. Download the MXNet source code from [GitHub](https://github.com/apache/incubator-mxnet). Don't forget to pull the submodules:
```
    git clone https://github.com/apache/incubator-mxnet.git --recursive
```

2. Copy file `3rdparty/mkldnn/config_template.vcxproj` to incubator-mxnet root.

3. Start a Visual Studio command prompt.

4. Use [CMake](https://cmake.org/) to create a Visual Studio solution in ```./build``` or some other directory. Make sure to specify the architecture in the 
[CMake](https://cmake.org/) command:
```
    mkdir build
    cd build
    cmake -G "Visual Studio 14 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release
```

5. In Visual Studio, open the solution file,```.sln```, and compile it.
These commands produce a library called ```libmxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.
Also ```libmkldnn.dll``` with be in the ```./build/3rdparty/mkldnn/src/Release/```

6. Make sure that all the dll files used above(such as `libmkldnn.dll`, `libmklml.dll`, `libiomp5.dll`, `libopenblas.dll`, etc) are added to the system PATH. For convinence, you can put all of them to ```\windows\system32```. Or you will come across `Not Found Dependencies` when loading mxnet.

<h2 id="4">Verify MXNet with python</h2>

```
export PYTHONPATH=~/incubator-mxnet/python
pip install --upgrade pip 
pip install --upgrade jupyter graphviz cython pandas bokeh matplotlib opencv-python requests
python -c "import mxnet as mx;print((mx.nd.ones((2, 3))*2).asnumpy());"

Expected Output:

[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

### Verify whether MKLDNN works:

```
WIP
```

<h2 id="5">Build/Install MXNet with a full MKL installation</h2>

To make it convenient for customers, Intel introduced a new license called [Intel® Simplified license](https://software.intel.com/en-us/license/intel-simplified-software-license) that allows to redistribute not only dynamic libraries but also headers, examples and static libraries.

Installing and enabling the full MKL installation enables MKL support for all operators under the linalg namespace.

  1. Download and install the latest full MKL version following instructions on the [intel website.](https://software.intel.com/en-us/mkl)

  2. Run 'make -j ${nproc} USE_BLAS=mkl'

  3. Navigate into the python directory

  4. Run 'sudo python setup.py install'