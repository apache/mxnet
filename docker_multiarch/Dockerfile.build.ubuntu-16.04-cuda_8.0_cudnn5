FROM nvidia/cuda:8.0-cudnn5-devel

RUN apt-get update &&\
    apt-get install -y wget python3.5 gcc-4.9 gcc-5 g++-4.9 g++-5 cmake less python3-pip python3-dev\
    build-essential git pkgconf\
    libopenblas-dev liblapack-dev\
    maven default-jdk\
    &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /work
#RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
ADD mxnet mxnet

# Compile MxNet
ENV BUILD_OPTS "USE_OPENCV=0 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1"
WORKDIR /work/mxnet
RUN make -j$(nproc) $BUILD_OPTS

WORKDIR /work/build/
RUN cp /work/mxnet/lib/* .

# Scala packag
#WORKDIR /work
#RUN wget --quiet http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb
#RUN dpkg -i scala-2.11.8.deb && rm scala-2.11.8.deb

#WORKDIR /work/mxnet
#RUN make scalapkg $BUILD_OPTS

#WORKDIR /work/build/scala_gpu
#RUN cp /work/mxnet/scala-package/assembly/linux-x86_64-gpu/target/*.jar .
