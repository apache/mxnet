# Before building this image you would need to build MXNet by executing:
# docker build -f Dockerfile.build.ubuntu-17.04 -t mxnet.build.ubuntu-17.04 .
# if you haven't done it before.

FROM mxnet.build.ubuntu-17.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y\
    libmouse-perl pdl cpanminus swig libgraphviz-perl
RUN rm -rf /var/lib/apt/lists/*

RUN cpanm -q Function::Parameters

WORKDIR /work/mxnet/perl-package/AI-MXNetCAPI
RUN perl Makefile.PL && make install

WORKDIR /work/mxnet/perl-package/AI-NNVMCAPI/
RUN perl Makefile.PL && make install

WORKDIR /work/mxnet/perl-package/AI-MXNet/
RUN	perl Makefile.PL && make install
