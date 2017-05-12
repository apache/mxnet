# Installing MXNet on CentOS

**NOTE:** For MXNet with Python installation, please refer to the [new install guide](http://mxnet.io/get_started/install.html).

MXNet currently supports Python, R, Julia, Scala, and Perl. For users on CentOS with Docker environment, MXNet provides [Docker installation guide](http://mxnet.io/get_started/docker_setup.html). If you do not have a Docker environment set up, follow below-provided step by step instructions.


## Minimum Requirements
Make sure you have the root permission, and `yum` is properly installed. Check it using the following command:

```bash
sudo yum check-update
```
If you don't get an error message, then `yum` is installed.

**To install MXNet on CentOS, you must have the following:**

1. gcc, g++ (4.8 or later)
2. python2, python-numpy, python-pip, clang
3. graphviz, jupyter (pip or yum install)
4. OpenBLAS
5. CUDA for GPU
6. cmake and opencv (do not use yum to install opencv, some shared libs may not be installed)

## Install Dependencies
Make sure your machine is connected to Internet. A few installations need to download (`git clone` or `wget`) some packages from Internet.

### Install Basic Environment
```bash
	# Install gcc-4.8/make and other development tools
	sudo yum install -y gcc
	sudo yum install -y gcc-c++
	sudo yum install -y clang

	# Install Python, Numpy, pip and set up tools.
	sudo yum groupinstall -y "Development Tools"
	sudo yum install -y python27 python27-setuptools python27-tools python-pip
	sudo yum install -y python27-numpy

	# install graphviz, jupyter
	sudo pip install graphviz
	sudo pip install jupyter
```
### Install OpenBLAS
Note that OpenBLAS can be replaced by other BLAS libs, e.g, Intel MKL.

```bash
	# Install OpenBLAS at /usr/local/openblas
	git clone https://github.com/xianyi/OpenBLAS
	cd OpenBLAS
	make -j $(($(nproc) + 1))
	sudo make PREFIX=/usr/local install
	cd ..
```
### Install CUDA for GPU
Note: Setting up CUDA is optional for MXNet. If you do not have a GPU machine (or if you want to train with CPU), you can skip this section and proceed with installation of OpenCV.

If you plan to build with GPU, you need to set up the environment for CUDA and CUDNN.

First, download and install [CUDA 8 toolkit](https://developer.nvidia.com/cuda-toolkit).

Then download [cudnn 5](https://developer.nvidia.com/cudnn).

Unzip the file and change to the cudnn root directory. Move the header and libraries to your local CUDA Toolkit folder:

```bash
    tar xvzf cudnn-8.0-linux-x64-v5.1-ga.tgz
    sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    sudo ldconfig
```
### Install opencv
Note: Setting up opencv is optional but strongly recommended for MXNet, unless you do not want to work on Computer Vision and Image Augmentation. If you are quite sure about that, skip this section and  set `USE_OPENCV = 0` in `config.mk`.

The Open Source Computer Vision (OpenCV) library contains programming functions for computer vision and image augmentation. For more information, see [OpenCV](https://en.wikipedia.org/wiki/OpenCV).

```bash
	# Install cmake for building opencv
	sudo yum install -y cmake
	# Install OpenCV at /usr/local/opencv
	git clone https://github.com/opencv/opencv
	cd opencv
	mkdir -p build
	cd build
	cmake -D BUILD_opencv_gpu=OFF -D WITH_EIGEN=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
	sudo make PREFIX=/usr/local install
```

## Install MXNet

### Build MXNet shared library
After installing the dependencies, use the following command to pull the MXNet source code from GitHub.

```bash
    # Download MXNet source code to ~/mxnet directory
    git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive
    # Move to source code parent directory
    cd ~/mxnet
    cp make/config.mk .
    # Replace this line if you use other BLAS libs
    echo "USE_BLAS=openblas" >>config.mk
    echo "ADD_CFLAGS += -I/usr/include/openblas" >>config.mk
    echo "ADD_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs" >>config.mk
```

If building with ```GPU``` support, run below commands to add GPU dependency configurations to `config.mk` file:

```bash
    echo "USE_CUDA=1" >>config.mk
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
    echo "USE_CUDNN=1" >>config.mk
```

Then build mxnet:

```bash
    make -j$(nproc)
```

Executing these commands creates a library called ```libmxnet.so``` in `~/mxnet/lib/`.

### Install MXNet for R, Julia, Scala, and Perl.

- [R](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-r)
- [Julia](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-julia)
- [Scala](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-scala)
- [Perl](http://mxnet.io/get_started/amazonlinux_setup.html#install-the-mxnet-package-for-perl)

## Troubleshooting

Here is some information to help you troubleshoot, in case you encounter error messages:

**1. Cannot build opencv from source code**

This may be caused by download failure during building, e.g., `ippicv`.

Prepare some large packages by yourself, then copy them to the right place, e.g, `opencv/3rdparty/ippicv/downloads/linux-808XXXXXXXXX/`.

**2. Link errors when building MXNet**

```bash
/usr/bin/ld: /tmp/ccQ9qruP.o: undefined reference to symbol '_ZN2cv6String10deallocateEv'
/usr/local/lib/libopencv_core.so.3.2: error adding symbols: DSO missing from command line
```
This error occurs when you already have old opencv (e.g, 2.4) installed using `yum` (in `/usr/lib64`). When g++ tries to link opencv libs, it will first find and link old opencv libs in `/usr/lib64`.

Please modify `config.mk` in `mxnet` directory, and add `-L/usr/local/lib` to `ADD_CFLAGS`.

```bash
	ADD_CFLAGS += -I/usr/include/openblas -L/usr/local/lib
```
This solution solves this link error, but there are still lots of warnings.


## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
