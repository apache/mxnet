# -*- mode: dockerfile -*-
# dockerfile to build libmxnet.so for armv7
FROM dockcross/linux-armv7

RUN apt-get update && \
    apt-get install -y libopenblas-dev:armhf && \
    rm -rf /var/lib/apt/lists/*

ENV ARCH armv71
ENV CC /usr/bin/arm-linux-gnueabihf-gcc
ENV CXX /usr/bin/arm-linux-gnueabihf-g++
ENV BUILD_OPTS "USE_OPENCV=0 USE_BLAS=openblas USE_SSE=0"

# Build MXNet

WORKDIR /work
#ADD https://api.github.com/repos/apache/incubator-mxnet/git/refs/heads/master mxnet_version.json
#RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
ADD mxnet mxnet

WORKDIR /work/mxnet
ADD arm.crosscompile.mk make/config.mk
RUN make -j$(nproc) $BUILD_OPTS

WORKDIR /work/build/
RUN cp /work/mxnet/lib/* .
