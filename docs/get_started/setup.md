# Overview

You can run MXNet on Amazon Linux, Ubuntu/Debian, OS X, and Windows operating systems. MXNet can also be run on Docker and on Cloud like AWS. MXNet currently supports Python, R, Julia, Scala, and Perl programming languages.

Step by step instructions for setting up MXNet:

- [Installing MXNet on OS X (Mac)](http://mxnet.io/get_started/osx_setup.html)
- [Installing MXNet on Ubuntu](http://mxnet.io/get_started/ubuntu_setup.html)
- [Installing MXNet on Windows](http://mxnet.io/get_started/windows_setup.html)
- [Installing MXNet on Amazon Linux](http://mxnet.io/get_started/amazonlinux_setup.html)
- [Installing MXNet on CentOS](http://mxnet.io/get_started/centos_setup.html)
- [MXNet with Docker](http://mxnet.io/get_started/docker_setup.html)
- [Installing MXNet on the Cloud (AWS AMI)](http://mxnet.io/get_started/cloud_setup.html)
- [Installing MXNet on Raspberry Pi (Raspbian)](http://mxnet.io/get_started/raspbian_setup.html)

This topic also covers the following:
- [Prerequisites for using MXNet](#prerequisites)
- [Common installation problems](#common-installation-problems)
- [Build the Dependent Libraries from Source Code](#build-the-dependent-libraries-from-source-code)

If you encounter problems with the set up instructions and don't find a solution in [Common installation problems](#common-installation-problems), ask questions at [mxnet/issues](https://github.com/dmlc/mxnet/issues). If you can fix the problem, send a
pull request. For details, see [contribution guidelines](http://mxnet.io/community/index.html).

# Prerequisites

This section lists the basic requirements for running MXNet, requirements for using it with GPUs, and requirements to support computer vision and image augmentation.

**Note:**  Setting up all basic(CPU only) required dependencies is covered as part of individual OS installation guide. This section is provided for power users who want to set up alternate dependencies(GPU/Intel MKL etc..) and experiment.

## Minimum Requirements

You must have the following:

- A C++ compiler that supports C++ 11. The C++ compiler compiles and builds MXNet source code. Supported compilers include the following:

- [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/)

- [Clang](http://clang.llvm.org/)

- A [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic Linear Algebra Subprograms) library.
  BLAS libraries contain routines that provide the standard building blocks for performing basic vector and matrix operations. You need a BLAS library to perform basic linear algebraic operations. Supported BLAS libraries include the following:
  * [libblas](http://www.netlib.org/blas/)
  * [openblas](http://www.openblas.net/)
  * [Intel MKL](https://software.intel.com/en-us/node/528497)

- [Graphviz](http://www.graphviz.org/) for visualizing the network graphs.
- [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/) for running examples and tutorials of MXNet.

## Requirements for Using GPUs

* A GPU with support for Compute Capability 2.0 or higher.
  Compute capability describes which features are supported by CUDA hardware. For a list of features supported by each compute capability, see [CUDA Version features and specifications](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications). For a list of features supported by NVIDIA GPUs, see [CUDA GPUs](https://developer.nvidia.com/cuda-gpus).
* The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 7.0 or higher.
  The CUDA Toolkit is an environment that allows MXNet to run on NVIDIA GPUs. It includes a compiler, math libraries, and debugging tools. To download the latest version, see [CUDA Toolkit download page](https://developer.nvidia.com/cuda-toolkit).
* The CuDNN (CUDA Deep Neural Network) library.
  The CuDNN library accelerates GPU computation by providing low-level GPU performance tuning. To download the latest version, see [CUDA Deep Neural Network](https://developer.nvidia.com/cudnn).

## Requirements for Computer Vision and Image Augmentation Support

If you need to support computer vision and image augmentation, you need
[OpenCV](http://opencv.org/).
The Open Source Computer Vision (OpenCV) library contains programming functions for computer vision and image augmentation. For more information, see [OpenCV](https://en.wikipedia.org/wiki/OpenCV).

# Build the Dependent Libraries from Source Code
This section provides instructions on how to build MXNet's dependent libraries from source code. This approach is useful in two specific situations:

- If you are using an earlier version of Linux on your server and required packages are either missing or Yum or apt-get didn't install a later version of the packages.

- If you do not have root permission to install packages. In this case, you need to change the installation directory from /usr/local to one where you do have permission. The following examples use the directory ${HOME}.

## Building GCC from Source Code
To build the GNU Compiler Collection (GCC) from source code, you need the 32-bit libc library.

1. Install libc with one of the following system-specific commands:

	```bash
		sudo apt-get install libc6-dev-i386 # In Ubuntu
		sudo yum install glibc-devel.i686   # In RHEL (Red Hat Linux)
		sudo yum install glibc-devel.i386   # In CentOS 5.8
		sudo yum install glibc-devel.i686   # In CentOS 6/7
	```
2. To download and extract the GCC source code, run the following commands:

	```bash
		wget http://mirrors.concertpass.com/gcc/releases/gcc-4.8.5/gcc-4.8.5.tar.gz
		tar -zxf gcc-4.8.5.tar.gz
		cd gcc-4.8.5
		./contrib/download_prerequisites
	```
3. To build GCC, run the following commands:

	```bash
		mkdir release && cd release
		../configure --prefix=/usr/local --enable-languages=c,c++
		make -j10
		sudo make install
	```
4. Add the lib path to your ~/.bashrc file:

	```bash
		export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
	```
## Build OpenCV from Source Code
To build OpenCV from source code, you need the ```cmake``` library .

* If you don't have cmake or if your version of cmake is earlier than 3.6.1 (such as the default version of cmake on RHEL), run the following commands to install a newer version of cmake:

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

* Build OpenCV. The following commands build OpenCV without GPU support, which might significantly slow down an MXNet program running on a GPU processor. It also disables 1394 which might generate a warning:

	```bash
		cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
		make -j8
		sudo make install
	```
* Add the lib path to your ```~/.bashrc``` file:

	```bash
		export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
	```
# Common Installation Problems
This section provides solutions for common installation problems.
## General
**Message:** ImportError: No module named _graphviz

**Cause:** Graphviz is not installed.

**Solution:**
On Mac, You can install Graphviz with below command
```bash
  brew install graphviz
```
Or, using pip
```bash
  brew install python
  pip install graphviz
```
**Message:** RuntimeError: failed to execute ['dot', '-Tsvg'], make sure the Graphviz executables are on your systems' path

**Cause:** Graphviz executable (lib) path is currently not in the system path and program is unable to use Graphviz for plotting the graph

**Solution:** Add Graphviz executable (lib) path to your system path.
On Mac/Linux machines, Graphviz is generally installed in - ```/usr/local/lib/graphviz/``` or ```/usr/lib/graphviz/``` or ```/usr/lib64/graphviz/``` and on Windows - ```C:\Program Files (x86)\Graphviz2.38\bin```.

**Note** If you are using Jupyter notebook, you may need to restart the kernel to refresh the system path and find Graphviz executable.

## Mac OS X Error Message
**Message:** link error ld: library not found for -lgomp

**Cause:** The GNU implementation of OpenMP is not in the operating system's library path.

**Solution:** Add the location of OpenMP to the library path:

* Create the locate database by running the following command:

	```bash
		sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.locate.plist
	```
* Locate the OpenMP library by running the following command:

	```bash
		locate libgomp.dylib
	```
* To add OpenMP to your library path, replace ```path1``` in the following command with the output from the last command:

	```bash
		ln -s path1 /usr/local/lib/libgomp.dylib
	```

* To build your changes, run the following command:

	```bash
		make -j$(sysctl -n hw.ncpu)
	```
## R Error Message
**Message** Unable to load mxnet after enabling CUDA

**Solution:** If you enabled CUDA when you installed MXNet, but can't load mxnet, add the following lines to your ```$RHOME/etc/ldpaths``` environment variable:

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

**Note:** To find your $RHOME environment variable, use the ```R.home()``` command in R.

# Next Steps
* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
