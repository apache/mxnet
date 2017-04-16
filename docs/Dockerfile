FROM ubuntu:14.04
MAINTAINER Mu Li <muli@cs.cmu.edu>

#
# First, build MXNet binaries (ref mxnet/docker/cpu/Dockerfile)
#

RUN apt-get update && apt-get install -y build-essential git libopenblas-dev libopencv-dev
RUN git clone --recursive https://github.com/dmlc/mxnet/ && cd mxnet && \
    cp make/config.mk . && \
    echo "USE_BLAS=openblas" >>config.mk && \
    make -j$(nproc)

# python pakcage
RUN apt-get install -y python-numpy wget unzip
ENV PYTHONPATH /mxnet/python

#
# Now set up tools for doc build
#

RUN apt-get update && apt-get install -y \
    doxygen \
    build-essential \
    git \
    python-pip

RUN pip install sphinx==1.3.5 CommonMark==0.5.4 breathe mock==1.0.1 recommonmark

WORKDIR /opt/mxnet/docs

# Fool it into thinking it's on a READTHEDOCS server, so it builds the
# API reference
ENV READTHEDOCS true

ENTRYPOINT /opt/mxnet/docs/build-preview.sh

EXPOSE 8008

# Put this at the end so that you don't have to rebuild the earlier
# layers when iterating on the docs themselves.
ADD . /opt/mxnet/docs

