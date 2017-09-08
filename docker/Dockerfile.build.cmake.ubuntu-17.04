FROM ubuntu:17.04


RUN apt-get update &&\
    apt-get install -y wget python3.5 gcc-4.9 gcc-5 g++-4.9 g++-5 cmake less python3-pip python3-dev\
    build-essential git pkgconf\
    libopenblas-dev liblapack-dev\
    maven default-jdk\
    ninja-build\
    libgtest-dev\
    &&\
    rm -rf /var/lib/apt/lists/*



###########################
# Build gtest
WORKDIR /work/googletest
RUN cmake /usr/src/googletest/googletest/ -GNinja
RUN ninja
RUN cp libgtest.a /usr/lib
###########################



WORKDIR /work
#RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
ADD mxnet mxnet

WORKDIR mxnet/build
RUN cmake -DUSE_CUDA=OFF -DUSE_OPENCV=OFF -GNinja .. 
RUN ninja


# Copy artifacts
RUN mkdir -p /work/build
RUN cp *.a *.so /work/build
