# -*- mode: dockerfile -*-
# dockerfile to build libmxnet.so for armv7
FROM dockcross/linux-arm64

ENV ARCH aarch64
ENV BUILD_OPTS "USE_BLAS=openblas USE_SSE=0 USE_OPENCV=0"
ENV CC /usr/bin/aarch64-linux-gnu-gcc
ENV CXX /usr/bin/aarch64-linux-gnu-g++
ENV FC /usr/bin/aarch64-linux-gnu-gfortran-4.9
ENV HOSTCC gcc

WORKDIR /work

# Build OpenBLAS
ADD https://api.github.com/repos/xianyi/OpenBLAS/git/refs/heads/master /tmp/openblas_version.json
RUN git clone https://github.com/xianyi/OpenBLAS.git && \
    cd OpenBLAS && \
    make -j$(nproc) TARGET=ARMV8 && \
    make install && \
    ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib/libopenblas.so && \
    ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/libopenblas.a && \
    ln -s /opt/OpenBLAS/lib/libopenblas.a /usr/lib/liblapack.a

ENV LD_LIBRARY_PATH /opt/OpenBLAS/lib
ENV CPLUS_INCLUDE_PATH /opt/OpenBLAS/include

# Build MXNet
#ADD https://api.github.com/repos/apache/incubator-mxnet/git/refs/heads/master mxnet_version.json
#RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
ADD mxnet mxnet

WORKDIR /work/mxnet
ADD arm.crosscompile.mk make/config.mk
RUN make -j$(nproc) $BUILD_OPTS

WORKDIR /work/build/
RUN cp /work/mxnet/lib/* .
