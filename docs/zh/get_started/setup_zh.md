# 概述

你可以在 Amazon Linux, Ubuntu/Debian, OS X, and Windows 操作系统上运行 MXNet,同时 Docker 和云服务(比如AWS)也可以运行。MXNet目前支持 Python, R, Julia 和 Scala 编程语言

配置 MXNet 详细教程:

- [MXNet with Docker](./docker_setup_zh.md)
- [云服务器(AWS AMI)安装](./cloud_setup_zh.md)
- [Ubuntu安装](./ubuntu_setup_zh.md)
- [Amazon Linux安装](./amazonlinux_setup_zh.md)
- [OS X (Mac)安装](./osx_setup_zh.md)
- [Windows安装](./windows_setup_zh.md)

本文还包含如下:
- [MXNet 设备要求](#设备要求)
- [常见安装问题](#常见安装问题)
- [编译依赖库](#编译依赖库)

如果你在安装中碰到问题，并在在[常见安装问题](#常见安装问题)中找不到解决办法，可以在 [mxnet/issues](https://github.com/dmlc/mxnet/issues) 发起提问.如果你可以解决这个问题，可以发起 pull request。细节可以参考 [contribution guidelines](http://mxnet.io/community/index.html).


# 设备要求

这里列出了运行 MXNet 的基本要求、在 GPU 上运行的要求、计算机视觉和图像增强的要求。

**注意:**  配置所有基本(仅 CPU)要求在各自的操作系统安装指南中已经说明。这里只提供给配置交替依赖(alternate dependencies)(GPU/Intel MKL 等等..)和实验的高级用户.

## 最低要求

你必须具备如下条件:

- 一个支持 C++ 11 的编译器。此编译器用来编译 MXNet 的源码。支持的编译器如下:
  * [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/)
  * [Clang](http://clang.llvm.org/)

- 一个 [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic Linear Algebra Subprograms) 库。BLAS 库包含了处理向量和矩阵运算的标准编译模块。你需要 BLAS 库来处理线性代数运算。支持的 BLAS 库包含如下:
  * [libblas](http://www.netlib.org/blas/)
  * [openblas](http://www.openblas.net/)
  * [Intel MKL](https://software.intel.com/en-us/node/528497)

- [Graphviz](http://www.graphviz.org/) 用来查看图像.
- [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/) 用来运行例子和教程.

## 使用 GPU 的要求

* 一个计算能力(Compute Capability)2.0及以上的 GPU。计算能力(Compute Capability)是 CUDA 硬件的特性。每种计算能力的详细特性可以参考列表 [CUDA Version features and specifications](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications)。 NVIDIA GPU 支持的特性，可以参考列表:[CUDA GPUs](https://developer.nvidia.com/cuda-gpus).
* [CUDA工具包](https://developer.nvidia.com/cuda-toolkit) 7.0 及以上。CUDA 工具包是一个可以让 MXNet 运行在 NVIDIA GPU 上的环境，它包含了一个编译器，一个数学库和一个调试工具。参考 [CUDA Toolkit download page](https://developer.nvidia.com/cuda-toolkit) 下载最新版。
* CuDNN (CUDA Deep Neural Network) 库。CuDNN 库提供调节底层GPU性能的方法，用来调节 GPU 的计算能力。参考 [CUDA Deep Neural Network](https://developer.nvidia.com/cudnn) 下载最新版。

## 计算机视觉和图像增强的要求

如果你需要支持计算机视觉和图像增强，需要安装 [OpenCV](http://opencv.org/).Open Source Computer Vision (OpenCV)库提供了计算机视觉和图像增强所需要的功能，详细介绍参考 [OpenCV](https://en.wikipedia.org/wiki/OpenCV).

# 编译依赖库

这里介绍如何从源码编译 MXNet 的依赖库。这个方法在两种特殊环境中很有用:

- 如果你的服务器正在使用较早版本的 Linux，依赖的包已经找不到，或者 Yum apt-get 不能安装新版本。

- 如果你没有 root 权限来安装依赖包。你需要把安装目录从 /usr/local 移动到有权限的目录下。下面的例子是使用 ${HOME} 目录。

## 从源码编译 GCC
你需要 32-bit libc 库，来编译 GNU Complier Collection (GCC)。

1. 根据操作系统选择一条命令来安装 libc:

	```bash
		sudo apt-get install libc6-dev-i386 # In Ubuntu
		sudo yum install glibc-devel.i686   # In RHEL (Red Hat Linux)
		sudo yum install glibc-devel.i386   # In CentOS 5.8
		sudo yum install glibc-devel.i686   # In CentOS 6/7
	```
2. 通过下面的命令下载 GCC 源码:

	```bash
		wget http://mirrors.concertpass.com/gcc/releases/gcc-4.8.5/gcc-4.8.5.tar.gz
		tar -zxf gcc-4.8.5.tar.gz
		cd gcc-4.8.5
		./contrib/download_prerequisites
	```
3. 通过下面命令编译 GCC:

	```bash
		mkdir release && cd release
		../configure --prefix=/usr/local --enable-languages=c,c++
		make -j10
		sudo make install
	```
4. 将库路径添加到 ```~/.bashrc```:

	```bash
		export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
	```
## 编译 OpenCV
你需要 ```cmake```库来编译 OpenCV。

* 如果你没有 cmake 或者 cmake 版本早于 3.6.1(比如 RHEL 默认的 cmake 版本)，通过下面的命令来安装新版 cmake:

	```bash
		wget https://cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.tar.gz
		tar -zxvf cmake-3.6.1-Linux-x86_64.tar.gz
		alias cmake="cmake-3.6.1-Linux-x86_64/bin/cmake"
	```

* 通过下面命令下载 OpenCV 源码:

	```bash
		wget https://codeload.github.com/opencv/opencv/zip/2.4.13
		unzip 2.4.13
		cd opencv-2.4.13
		mkdir release
		cd release/
	```

* 编译 openCV。下面的命令可以编译不支持 GPU 的 openCV，这可能意味着 MXNet 程序在 GPU 上运行会比较慢。它还可能发出警告的 1394 功能

	```bash
		cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
		make -j8
		sudo make install
	```
* 将库路径添加到 ```~/.bashrc```:

	```bash
		export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
	```
# 常见安装问题
这里介绍常见安装问题的解决办吧
## 普通
**消息:** ImportError: No module named _graphviz

**原因:** Graphviz 未安装.

**解决:** 在 Mac 上，你可以通过下面命令在安装 Graphviz

```bash
  brew install graphviz
```
或者使用 pip
```bash
  brew install python
  pip install graphviz
```
**消息:** RuntimeError: failed to execute ['dot', '-Tsvg'], make sure the Graphviz executables are on your systems' path

**原因:** Graphviz 的可执行(库)路径不再当前系统的 path 中，程序无法使用 Graphviz 绘制图像

**解决:** 将 Graphviz 可执行(库)路径添加到系统 path 中。在 Mac/Linux 上，Graphviz 通常安装在 - ```/usr/local/lib/graphviz/``` or ```/usr/lib/graphviz/``` or ```/usr/lib64/graphviz/``` and on Windows - ```C:\Program Files (x86)\Graphviz2.38\bin```.

**注意** 如果你使用 Jupyter notebook,可能需要重启 kernel 来刷新系统 path。

## Mac OS X 错误消息
**消息:** link error ld: library not found for -lgomp

**原因:** OpenMP 不在系统库 path 中.

**解决:** 将 OpenMP 添加的系统库 path 中。:

* 用下面命令创建定位数据库(locate database):

	```bash
		sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.locate.plist
	```
* 用下面命令定位 OpenMP 库:

	```bash
		locate libgomp.dylib
	```
* 将 OpenMP 添加到系统库 path 中, 用上一条命令的输出结果替换下面命令中的 ```path1``` :

	```bash
		ln -s path1 /usr/local/lib/libgomp.dylib
	```

* 用下面的命令编译你刚刚的修改:

	```bash
		make -j$(sysctl -n hw.ncpu)
	```
## R 错误消息
**消息** Unable to load mxnet after enabling CUDA

**解决:** 如果在安装时你打开了 CUDA 功能，但是不能加载 mxnet。将这些行添加到你的```$RHOME/etc/ldpaths``` 环境变量中:

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

**注意:** R 中可以通过 ```R.home()``` 命令来查找 $RHOME 环境变量.

# 下一步
* [教程](http://mxnet.io/tutorials/index.html)
* [如何使用](http://mxnet.io/how_to/index.html)
* [架构设计](http://mxnet.io/architecture/index.html)
