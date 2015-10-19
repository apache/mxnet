Installation Guide
==================
This page gives the detail of how to install mxnet packages on various systems.
We tried to listed the detailed, but if the information on this page does not work for you.
Please ask questions at [mxnet/issues](https://github.com/dmlc/mxnet/issues), better still
if you have ideas to improve this page, please send a pull request!

Contents
--------
- [Build MXNet Library](#build-mxnet-library)
  - Introduces how to build the mxnet core library for all packages.
  - Supported platforms: linux, windows, osx
- [Advanced Build Configurations](#advanced-build-configuration)
  - Introduces how to build mxnet with advanced features such as HDFS/S3 support, CUDNN
- [Python Package Installation](#python-package-installation)
- [R Package Installation](#r-package-installation)

Build MXNet Library
-------------------
MXNet have a general runtime library that can be used by various packages such as python, R and Julia.
This section gives details about how to build the mxnet library.
- On Linux/OSX the target library will be ```libmxnet.so```
- On Windows the target libary is ```mxnet.dll```

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

### Linux

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

### OSX
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

### Windows

Firstly, we should make your Visual Studio 2013 support more C++11 features.

 - Download and install [Visual C++ Compiler Nov 2013 CTP](http://www.microsoft.com/en-us/download/details.aspx?id=41151).
 - Copy all files in `C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP` (or the folder where you extracted the zip archive) to `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC` and overwrite all existed files. Don't forget to backup the original files before copying.

Secondly, fetch the third-party libraries, including [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download), [CuDNN](https://developer.nvidia.com/cudnn) and [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/)(ignore this if you have MKL).

 - NOTICE: You need to register as a NVIDIA community user to get the download link of CuDNN.

Finally, use CMake to create a Visual Studio solution in `./build/`. During configuration, you may need to set the path of each third-party library, until no error is reported. Open the solution and compile, you will get a `mxnet.dll` in `./build/Release` or `./build/Debug`.

Then proceed to package installation instructions for python or R in this page.

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
To install the R package. First finish the [Build MXNet Library](#build-mxnet-library) step.
Then use the following command to install mxnet at root folder

```bash
R CMD INSTALL R-Package
```

Hopefully, we will now have mxnet on R!

## Note on Library Build
We isolate the library build with Rcpp end to maximize the portability
  - MSVC is needed on windows to build the mxnet library, because of CUDA compatiblity issue of toolchains.
