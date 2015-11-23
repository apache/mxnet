Installation Guide
==================
This page gives the detail of how to install mxnet packages on various systems.
We tried to listed the detailed, but if the information on this page does not work for you.
Please ask questions at [mxnet/issues](https://github.com/dmlc/mxnet/issues), better still
if you have ideas to improve this page, please send a pull request!

Contents
--------
- [Building MXNet Library](#build-mxnet-library)
  - [Prerequisites](#prerequisites)
  - [Building on Linux](#building-on-linux)
  - [Building on OSX](#building-on-osx)
  - [Building on Windows](#building-on-windows)
  - [Installing pre-built packages on Windows](#installing-pre-built-packages-on-windows)
- [Advanced Build Configurations](#advanced-build-configuration)
  - Introduces how to build mxnet with advanced features such as HDFS/S3 support, CUDNN
- [Python Package Installation](#python-package-installation)
- [R Package Installation](#r-package-installation)
- [Docker Images](#docker-images)

Build MXNet Library
-------------------

### Prerequisites

MXNet have a general runtime library that can be used by various packages such as python, R and Julia.
This section gives details about how to build the mxnet library.
- On Linux/OSX the target library will be ```libmxnet.so```
- On Windows the target libary is ```libmxnet.dll```

Things to do before get started:

- Clone the project from github
```bash
git clone --recursive https://github.com/dmlc/mxnet
```

The system dependency requirement for mxnet libraries are

- Recent c++ compiler supporting C++ 11 such as `g++ >= 4.8`
- git
- BLAS library.
- opencv (optional if you do not need image augmentation, you can switch it off in config.mk)

### Building on Linux

On Ubuntu >= 13.10, one can install the dependencies by

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libblas-dev libopencv-dev
```

Then build mxnet on the project root
```bash
make -j4
```
Then proceed to package installation instructions for python or R in this page.

### Buillding on OSX
On OSX, we can install the dependencies by

```bash
brew update
brew tap homebrew/science
brew info opencv
brew install opencv
```

- Copy ```make/osx.mk``` to project root ```config.mk```.
```bash
cp make/osx.mk config.mk
```

Then build mxnet on the project root
```bash
make -j4
```

Then proceed to package installation instructions for python or R in this page.

### Building on Windows

Firstly, we should make your Visual Studio 2013 support more C++11 features.

 - Download and install [Visual C++ Compiler Nov 2013 CTP](http://www.microsoft.com/en-us/download/details.aspx?id=41151).
 - Copy all files in `C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP` (or the folder where you extracted the zip archive) to `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC` and overwrite all existed files. Don't forget to backup the original files before copying.

Secondly, fetch the third-party libraries, including [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download), [CuDNN](https://developer.nvidia.com/cudnn) and [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/)(ignore this if you have MKL).

 - NOTICE: You need to register as a NVIDIA community user to get the download link of CuDNN.

Finally, use CMake to create a Visual Studio solution in `./build/`. During configuration, you may need to set the path of each third-party library, until no error is reported. Open the solution and compile, you will get a `mxnet.dll` in `./build/Release` or `./build/Debug`.

Then proceed to package installation instructions for python or R in this page.

### Installing pre-built packages on Windows

Mxnet also provides pre-built packages on Windows. The pre-built package includes pre-build MxNet library, the dependent thrid-party libraries, a sample C++ solution in Visual Studio and the Python install script.

You can download the packages from the [Releases tab](https://github.com/dmlc/mxnet/releases) of MxNet. There are two variants provided: one with GPU support (using CUDA and CUDNN v3) and one without GPU support. You can choose one that fits your hardward configuration.

After download, unpack the package into a folder, say D:\MxNet, then install the package by double clicking the setupenv.cmd inside the folder. It will setup environmental variables needed by MxNet. After that, you should be able to usee the provided VS solution to build C++ programs, or to [install Python package](#python-package-installation).

Advanced Build Configurations
-----------------------------
The configuration of mxnet can be modified by ```config.mk```
- modify the compiling options such as compilers, CUDA, CUDNN, Intel MKL,
various distributed filesystem such as HDFS/Amazon S3/...
- First copy [make/config.mk](../make/config.mk) to the project root, then
  modify the according flags.

Python Package Installation
---------------------------
To install the python package. First finish the [Build MXNet Library](#build-mxnet-library) step.
Then use the following command to install mxnet.

```bash
cd python; python setup.py install
```

If anything goes well, now we can train a multilayer perceptron on the hand
digit recognition dataset.

```bash
cd ..; python example/mnist/mlp.py
```

YOu can also install python to your user directory instead of root.

```bash
cd python; python setup.py develop --user
```

R Package Installation
----------------------

For Windows/Mac users, we provide pre-built binary package using CPU.
You can install weekly updated package directly in R console:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
```

To install the R package. First finish the [Build MXNet Library](#build-mxnet-library) step.
Then use the following command to install dependencies and build the package at root folder

```bash
Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
cd ..
make rpkg
```

Now you should have the R package as a tar.gz file and you can install it as a normal package by (the version number might be different)

```bash
R CMD INSTALL mxnet_0.5.tar.gz
```

## Note on Library Build
We isolate the library build with Rcpp end to maximize the portability
  - MSVC is needed on windows to build the mxnet library, because of CUDA compatiblity issue of toolchains.

Docker Images
-------------
Builds of MXNet are available as [Docker](https://www.docker.com/whatisdocker) images:
[MXNet Docker (CPU)](https://hub.docker.com/r/kaixhin/mxnet/) or [MXNet Docker (CUDA)](https://hub.docker.com/r/kaixhin/cuda-mxnet/).
These are updated on a weekly basis with the latest builds of MXNet. Examples of running bash in a Docker container
are as follows:

```bash
sudo docker run -it kaixhin/mxnet
sudo docker run -it --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 kaixhin/cuda-mxnet:7.0
```

For a guide to Docker, see the [official docs](https://docs.docker.com/userguide/). For more details on how to use the
MXNet Docker images, including requirements for CUDA support, consult the [source project](https://github.com/Kaixhin/dockerfiles).
