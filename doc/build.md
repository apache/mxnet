Build and Installation
======================

Minimal system requirement:

- recent c++ compiler supporting C++ 11 such as `g++ >= 4.8`
- git
- BLAS library.
- opencv

On Ubuntu >= 13.10, one can install them by

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libblas-dev libopencv-dev
```

Then build mxnet

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; make -j4
```

To install the python package, first make sure `python >= 2.7` and `numpy >= ?` are installed, then

```bash
cd python; python setup.py install
```

If anything goes well, now we can train a multilayer perceptron on the hand
digit recognition dataset.

```bash
cd ..; python example/mnist/mlp.py
```

Advanced Build
--------------

- update the repo:

```bash
git pull
git submodule update
```

- install python package in developing model,

```bash
cd python; python setup.py develop --user
```

- modify the compiling options such as compilers, CUDA, CUDNN, Intel MKL,
various distributed filesystem such as HDFS/Amazon S3/...

  First copy [make/config.mk](../make/config.mk) to the project root, then
  modify the according flags.

Build in Visual Studio 2013
---------------------------

Firstly, we should make your Visual Studio 2013 support more C++11 features.

 - Download and install [Visual C++ Compiler Nov 2013 CTP](http://www.microsoft.com/en-us/download/details.aspx?id=41151). 
 - Copy all files in `C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP` (or the folder where you extracted the zip archive) to `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC` and overwrite all existed files. Don't forget to backup the original files before copying.

Secondly, fetch the third-party libraries, including [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download), [CuDNN](https://developer.nvidia.com/cudnn) and [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/)(ignore this if you have MKL).

 - NOTICE: You need to register as a NVIDIA community user to get the download link of CuDNN.

Finally, use CMake to create a Visual Studio solution in `./build/`. During configuration, you may need to set the path of each third-party library, until no error is reported. Open the solution and compile, you will get a `mxnet.dll` in `./build/Release` or `./build/Debug`.

The following steps are the same with Linux.
