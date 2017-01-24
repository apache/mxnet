#!/usr/bin/env bash

# Assuming the script is run at mxnet/tools/pip_package
# This script builds from scratch the dependencies of mxnet into static
# librareis and statically links them to produce a (mostly) standalone
# libmxnet.so, then packages it into the python wheel.
# It assumes the build environment to be a sandbox that doesn't have the .so
# objects for the dependencies, i.e. zlib, openblas, libjpeg, libpng, libtiff
# and opencv.

# Install necessary build tools
if [ -n "$(command -v apt-get)" ]; then
    sudo apt-get update;
    sudo apt-get install -y build-essential git python-pip zip pkg-config cmake
elif [ -n "$(command -v yum)" ]; then
    sudo yum install -y cmake
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y python27 python27-setuptools python27-tools python-pip
else
    echo "Need a package manager to install build tools, e.g. apt/yum"
    exit 1
fi
sudo pip install -U pip setuptools wheel

# Setup path to dependencies
export PKG_CONFIG_PATH=$HOME/lib/pkgconfig:$HOME/lib64/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$HOME/include:$CPATH

# Position Independent code must be turned on for statically linking .a
export CC="gcc -fPIC"
export CXX="g++ -fPIC"

# Dependencies can be updated here. Be sure to verify the download link before
# changing. The dependencies are:
ZLIB_VERSION=1.2.6
OPENBLAS_VERSION=0.2.19
JPEG_VERSION=8.4.0
PNG_VERSION=1.5.10
TIFF_VERSION=3.8.2
OPENCV_VERSION=2.4.13

# Download and build zlib
curl -L https://github.com/LuaDist/zlib/archive/$ZLIB_VERSION.zip -o $HOME/zlib.zip
unzip $HOME/zlib.zip -d $HOME
mkdir $HOME/zlib-$ZLIB_VERSION/build
cd $HOME/zlib-$ZLIB_VERSION/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$HOME \
      -D BUILD_SHARED_LIBS=OFF ..
make -j$(nproc)
make install
cd -

# download and build openblas
curl -L https://github.com/xianyi/OpenBLAS/archive/v$OPENBLAS_VERSION.zip -o $HOME/openblas.zip
unzip $HOME/openblas.zip -d $HOME
cd $HOME/OpenBLAS-$OPENBLAS_VERSION
make FC=gfortran -j $(($(nproc) + 1))
make PREFIX=$HOME install
cd -
ln -s $HOME/lib/libopenblas_haswellp-r0.2.19.a $HOME/lib/libcblas.a

# download and build libjpeg
curl -L https://github.com/LuaDist/libjpeg/archive/$JPEG_VERSION.zip -o $HOME/libjpeg.zip
unzip $HOME/libjpeg.zip -d $HOME
cd $HOME/libjpeg-$JPEG_VERSION
./configure --disable-shared --prefix=$HOME
make -j$(nproc)
make test
make install
cd -

# download and build libpng
curl -L https://github.com/LuaDist/libpng/archive/$PNG_VERSION.zip -o $HOME/libpng.zip
unzip $HOME/libpng.zip -d $HOME
mkdir $HOME/libpng-$PNG_VERSION/build
cd $HOME/libpng-$PNG_VERSION/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$HOME \
      -D PNG_CONFIGURE_LIBPNG=-fPIC \
      -D BUILD_SHARED_LIBS=OFF ..
make -j$(nproc)
make install
cd -

# download and build libtiff
curl -L https://github.com/LuaDist/libtiff/archive/$TIFF_VERSION.zip -o $HOME/libtiff.zip
unzip $HOME/libtiff.zip -d $HOME
cd $HOME/libtiff-$TIFF_VERSION
./configure --disable-shared --prefix=$HOME
make -j$(nproc)
make install
cd -

# download and build opencv since we need the static library
curl -L https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.zip -o $HOME/opencv.zip
unzip $HOME/opencv.zip -d $HOME
mkdir $HOME/opencv-$OPENCV_VERSION/build
cd $HOME/opencv-$OPENCV_VERSION/build
cmake -D WITH_1394=OFF \
      -D WITH_AVFOUNDATION=OFF \
      -D WITH_CUDA=OFF \
      -D WITH_VTK=OFF \
      -D WITH_CUFFT=OFF \
      -D WITH_CUBLAS=OFF \
      -D WITH_NVCUVID=OFF \
      -D WITH_EIGEN=ON \
      -D WITH_VFW=OFF \
      -D WITH_FFMPEG=OFF \
      -D WITH_GSTREAMER=OFF \
      -D WITH_GTK=OFF \
      -D WITH_JASPER=OFF \
      -D WITH_JPEG=ON \
      -D WITH_PNG=ON \
      -D WITH_QUICKTIME=OFF \
      -D WITH_TBB=ON \
      -D WITH_TIFF=OFF \
      -D WITH_V4L=OFF \
      -D WITH_LIBV4L=OFF \
      -D WITH_DSHOW=OFF \
      -D WITH_MSMF=OFF \
      -D WITH_OPENCL=OFF \
      -D WITH_OPENCLAMDFFT=OFF \
      -D WITH_OPENCLAMDBLAS=OFF \
      -D BUILD_SHARED_LIBS=OFF \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_gpu=OFF \
      -D BUILD_opencv_video=OFF \
      -D BUILD_opencv_contrib=OFF \
      -D BUILD_opencv_nonfree=OFF \
      -D BUILD_opencv_flann=OFF \
      -D BUILD_opencv_features2d=OFF \
      -D BUILD_opencv_calib3d=OFF \
      -D BUILD_opencv_objdetect=OFF \
      -D BUILD_opencv_ml=OFF \
      -D BUILD_opencv_photo=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_PACKAGE=OFF \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$HOME ..
make -j $(nproc)
make install # user will always have access to home, so no sudo needed
cd -

# Although .so building is explicitly turned off for most libraries, sometimes
# they still get created. So, remove them just to make sure they don't
# interfere, or otherwise we might get libmxnet.so that is not self-contained.
rm $HOME/{lib,lib64}/*.{so,so.0}

# Go to the parent path and build mxnet
cd ../../
BUILD_PIP_WHEEL=1 make -j $(nproc)

# Generate wheel. The output is in the mxnet/tools/pip_package/dist path.
cd tools/pip_package
python setup.py bdist_wheel
