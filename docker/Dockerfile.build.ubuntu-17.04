#
# Base image to build MXNet from source in ubuntu
#
# Other images depend on it, so build it like:
#
# docker build -f Dockerfile.build.ubuntu-17.04 -t mxnet.build.ubuntu-17.04 .
#
FROM ubuntu:17.04


RUN apt-get update &&\
    apt-get install -y wget python3.5 gcc-4.9 gcc-5 g++-4.9 g++-5 cmake less python3-pip python3-dev\
    build-essential git pkgconf\
    libopenblas-dev liblapack-dev\
    maven default-jdk

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /work
#RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
ADD mxnet mxnet

# Compile MxNet
ENV BUILD_OPTS "USE_OPENCV=0 USE_BLAS=openblas"
WORKDIR /work/mxnet
RUN make -j$(nproc) $BUILD_OPTS

# Copy artifacts
WORKDIR /work/build/
RUN cp /work/mxnet/lib/* .
