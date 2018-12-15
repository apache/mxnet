#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# This script builds the static library of opencv that can be used as dependency of mxnet.
# It expects openblas, libjpeg, libpng, libtiff, eigen, etc., to be in $DEPS_PATH.
OPENCV_VERSION=3.4.2
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
if [[ $PLATFORM == 'linux' ]]; then
    OPENCV_LAPACK_OPTIONS=" \
          -D OpenBLAS_HOME=$DEPS_PATH \
          -D OpenBLAS_INCLUDE_DIR=$DEPS_PATH/include \
          -D OpenBLAS_LIB=$DEPS_PATH/lib/libopenblas.a \
          -D LAPACK_INCLUDE_DIR=$DEPS_PATH/include \
          -D LAPACK_LINK_LIBRARIES=$DEPS_PATH/lib/ \
          -D LAPACK_LIBRARIES=$DEPS_PATH/lib/libopenblas.a \
          -D LAPACK_CBLAS_H='cblas.h' \
          -D LAPACK_LAPACKE_H='lapacke.h' \
          -D LAPACK_IMPL='OpenBLAS' \
          -D HAVE_LAPACK=1"
fi

if [[ ! -f $DEPS_PATH/lib/libopencv_core.a ]] || [[ ! -f $DEPS_PATH/lib/libopencv_imgcodecs.a ]] || [[ ! -f $DEPS_PATH/lib/libopencv_imgproc.a ]]; then
    # download and build opencv since we need the static library
    >&2 echo "Building opencv..."
    curl -s -L https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip -o $DEPS_PATH/opencv.zip
    unzip -q $DEPS_PATH/opencv.zip -d $DEPS_PATH
    mkdir -p $DEPS_PATH/opencv-$OPENCV_VERSION/build
    cd $DEPS_PATH/opencv-$OPENCV_VERSION/build
    cmake \
          -D OPENCV_ENABLE_NONFREE=OFF \
          -D WITH_1394=OFF \
          -D WITH_ARAVIS=OFF \
          -D WITH_AVFOUNDATION=OFF \
          -D WITH_CAROTENE=OFF \
          -D WITH_CLP=OFF \
          -D WITH_CSTRIPES=OFF \
          -D WITH_CPUFEATURES=OFF \
          -D WITH_CUBLAS=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_CUFFT=OFF \
          -D WITH_DIRECTX=OFF \
          -D WITH_DSHOW=OFF \
          -D WITH_EIGEN=ON \
          -D WITH_FFMPEG=OFF \
          -D WITH_GDAL=OFF \
          -D WITH_GDCM=OFF \
          -D WITH_GIGEAPI=OFF \
          -D WITH_GPHOTO2=OFF \
          -D WITH_GSTREAMER=OFF \
          -D WITH_GSTREAMER_0_10=OFF \
          -D WITH_GTK=OFF \
          -D WITH_GTK_2_X=OFF \
          -D WITH_HALIDE=OFF \
          -D WITH_IMAGEIO=OFF \
          -D WITH_IMGCODEC_HDR=OFF \
          -D WITH_IMGCODEC_PXM=OFF \
          -D WITH_IMGCODEC_SUNRASTER=OFF \
          -D WITH_INF_ENGINE=OFF \
          -D WITH_INTELPERC=OFF \
          -D WITH_IPP=OFF \
          -D WITH_IPP_A=OFF \
          -D WITH_ITT=OFF \
          -D WITH_JASPER=OFF \
          -D WITH_JPEG=ON \
          -D WITH_LAPACK=ON \
          -D WITH_LIBREALSENSE=OFF \
          -D WITH_LIBV4L=OFF \
          -D WITH_MATLAB=OFF \
          -D WITH_MFX=OFF \
          -D WITH_MSMF=OFF \
          -D WITH_NVCUVID=OFF \
          -D WITH_OPENCL=OFF \
          -D WITH_OPENCLAMDBLAS=OFF \
          -D WITH_OPENCLAMDFFT=OFF \
          -D WITH_OPENCL_SVM=OFF \
          -D WITH_OPENEXR=OFF \
          -D WITH_OPENGL=OFF \
          -D WITH_OPENMP=OFF \
          -D WITH_OPENNI=OFF \
          -D WITH_OPENNI2=OFF \
          -D WITH_OPENVX=OFF \
          -D WITH_PNG=ON \
          -D WITH_PROTOBUF=OFF \
          -D WITH_PTHREADS_PF=ON \
          -D WITH_PVAPI=OFF \
          -D WITH_QT=OFF \
          -D WITH_QTKIT=OFF \
          -D WITH_QUICKTIME=OFF \
          -D WITH_TBB=OFF \
          -D WITH_TIFF=ON \
          -D WITH_UNICAP=OFF \
          -D WITH_V4L=OFF \
          -D WITH_VA=OFF \
          -D WITH_VA_INTEL=OFF \
          -D WITH_VFW=OFF \
          -D WITH_VTK=OFF \
          -D WITH_WEBP=OFF \
          -D WITH_WIN32UI=OFF \
          -D WITH_XIMEA=OFF \
          -D WITH_XINE=OFF \
          -D BUILD_ANDROID_EXAMPLES=OFF \
          -D BUILD_ANDROID_PROJECTS=OFF \
          -D BUILD_ANDROID_SERVICE=OFF \
          -D BUILD_CUDA_STUBS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_FAT_JAVA_LIB=OFF \
          -D BUILD_IPP_IW=OFF \
          -D BUILD_ITT_IW=OFF \
          -D BUILD_JAVA=OFF \
          -D BUILD_JASPER=OFF \
          -D BUILD_JPEG=OFF \
          -D BUILD_OPENEXR=OFF \
          -D BUILD_PACKAGE=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_PNG=OFF \
          -D BUILD_SHARED_LIBS=OFF \
          -D BUILD_TBB=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_TIFF=OFF \
          -D BUILD_WEBP=OFF \
          -D BUILD_WITH_DEBUG_INFO=OFF \
          -D BUILD_WITH_DYNAMIC_IPP=OFF \
          -D BUILD_WITH_STATIC_CRT=OFF \
          -D BUILD_ZLIB=OFF \
          -D BUILD_opencv_apps=OFF \
          -D BUILD_opencv_aruco=OFF \
          -D BUILD_opencv_calib3d=OFF \
          -D BUILD_opencv_contrib=OFF \
          -D BUILD_opencv_dnn=OFF \
          -D BUILD_opencv_features2d=OFF \
          -D BUILD_opencv_flann=OFF \
          -D BUILD_opencv_gpu=OFF \
          -D BUILD_opencv_gpuarithm=OFF \
          -D BUILD_opencv_gpubgsegm=OFF \
          -D BUILD_opencv_gpucodec=OFF \
          -D BUILD_opencv_gpufeatures2d=OFF \
          -D BUILD_opencv_gpufilters=OFF \
          -D BUILD_opencv_gpuimgproc=OFF \
          -D BUILD_opencv_gpulegacy=OFF \
          -D BUILD_opencv_gpuoptflow=OFF \
          -D BUILD_opencv_gpustereo=OFF \
          -D BUILD_opencv_gpuwarping=OFF \
          -D BUILD_opencv_highgui=OFF \
          -D BUILD_opencv_java=OFF \
          -D BUILD_opencv_js=OFF \
          -D BUILD_opencv_ml=OFF \
          -D BUILD_opencv_ml=OFF \
          -D BUILD_opencv_nonfree=OFF \
          -D BUILD_opencv_objdetect=OFF \
          -D BUILD_opencv_photo=OFF \
          -D BUILD_opencv_python=OFF \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=OFF \
          -D BUILD_opencv_superres=OFF \
          -D BUILD_opencv_video=OFF \
          -D BUILD_opencv_videoio=OFF \
          -D BUILD_opencv_videostab=OFF \
          -D BUILD_opencv_viz=OFF \
          -D BUILD_opencv_world=OFF \
          $OPENCV_LAPACK_OPTIONS \
          -D OPENCV_LIB_INSTALL_PATH=lib \
          -D OPENCV_INCLUDE_INSTALL_PATH=include \
          -D CMAKE_LIBRARY_PATH=$DEPS_PATH/lib \
          -D CMAKE_INCLUDE_PATH=$DEPS_PATH/include \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=$DEPS_PATH ..
    if [[ $PLATFORM == 'linux' ]]; then
        cp $DIR/patch/opencv_lapack.h ./
    fi
    make
    make install
    cd -
    # @szha: compatibility header
    cat $DEPS_PATH/include/opencv2/imgcodecs/imgcodecs_c.h >> $DEPS_PATH/include/opencv2/imgcodecs.hpp
fi
