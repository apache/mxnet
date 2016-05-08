FROM nvidia/cuda:7.5-cudnn4-devel
MAINTAINER Mu Li <muli@cs.cmu.edu>

# install the core library
RUN apt-get update && apt-get install -y build-essential git libopenblas-dev libopencv-dev
RUN git clone --recursive https://github.com/dmlc/mxnet/ && cd mxnet && \
    cp make/config.mk . && \
    echo "USE_CUDA=1" >>config.mk && \
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk && \
    echo "USE_CUDNN=1" >>config.mk && \
    echo "USE_BLAS=openblas" >>config.mk && \
    make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH 

# python pakcage
RUN apt-get install -y python-numpy wget unzip
ENV PYTHONPATH /mxnet/python
