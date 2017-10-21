# -*- mode: dockerfile -*-
FROM dockcross/base:latest
MAINTAINER Pedro Larroy "pllarroy@amazon.com"

# The cross-compiling emulator
RUN apt-get update && apt-get install -y \
  qemu-user \
  qemu-user-static \
  unzip

ENV CROSS_TRIPLE=arm-linux-androideabi
ENV CROSS_ROOT=/usr/${CROSS_TRIPLE}
ENV AS=${CROSS_ROOT}/bin/${CROSS_TRIPLE}-as \
    AR=${CROSS_ROOT}/bin/${CROSS_TRIPLE}-ar \
    CC=${CROSS_ROOT}/bin/${CROSS_TRIPLE}-gcc \
    CPP=${CROSS_ROOT}/bin/${CROSS_TRIPLE}-cpp \
    CXX=${CROSS_ROOT}/bin/${CROSS_TRIPLE}-g++ \
    LD=${CROSS_ROOT}/bin/${CROSS_TRIPLE}-ld

ENV ANDROID_NDK_REVISION 15c
RUN mkdir -p /build && \
    cd /build && \
    curl -O https://dl.google.com/android/repository/android-ndk-r${ANDROID_NDK_REVISION}-linux-x86_64.zip && \
    unzip ./android-ndk-r${ANDROID_NDK_REVISION}-linux-x86_64.zip && \
    cd android-ndk-r${ANDROID_NDK_REVISION} && \
    ./build/tools/make_standalone_toolchain.py \
      --stl=libc++ \
      --arch arm \
      --api 16 \
      --install-dir=${CROSS_ROOT} && \
    cd / && \
    rm -rf /build && \
    find ${CROSS_ROOT} -exec chmod a+r '{}' \; && \
    find ${CROSS_ROOT} -executable -exec chmod a+x '{}' \;


ENV DEFAULT_DOCKCROSS_IMAGE dockcross/android-arm

# COPY Toolchain.cmake ${CROSS_ROOT}/
# ENV CMAKE_TOOLCHAIN_FILE ${CROSS_ROOT}/Toolchain.cmake

# Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG IMAGE
ARG VCS_REF
ARG VCS_URL
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name=$IMAGE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url=$VCS_URL \
      org.label-schema.schema-version="1.0"

ENV CC /usr/arm-linux-androideabi/bin/arm-linux-androideabi-gcc
ENV CXX /usr/arm-linux-androideabi/bin/arm-linux-androideabi-g++

# Build OpenBLAS
# https://github.com/xianyi/OpenBLAS/wiki/How-to-build-OpenBLAS-for-Android
RUN git clone https://github.com/xianyi/OpenBLAS.git && \
    cd OpenBLAS && \
    make -j$(nproc) TARGET=ARMV7 ARM_SOFTFP_ABI=1 HOSTCC=gcc NOFORTRAN=1 libs

ENV OPENBLAS_ROOT /work/OpenBLAS
ENV LIBRARY_PATH /work/OpenBLAS/lib/:/work/OpenBLAS/:$LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH /work/OpenBLAS/include/:/work/OpenBLAS/:$CPLUS_INCLUDE_PATH
WORKDIR /work

ENV CC /usr/arm-linux-androideabi/bin/arm-linux-androideabi-clang
ENV CXX /usr/arm-linux-androideabi/bin/arm-linux-androideabi-clang++
ENV BUILD_OPTS "USE_BLAS=openblas USE_SSE=0 DMLC_LOG_STACK_TRACE=0 USE_OPENCV=0 USE_LAPACK=0"

# Build MXNet
ADD mxnet mxnet
ADD arm.crosscompile.android.mk /work/mxnet/make/config.mk
RUN cd mxnet && \
    make -j$(nproc) $BUILD_OPTS

WORKDIR /work/build/
RUN cp /work/mxnet/lib/* .
