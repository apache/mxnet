# Before building this image you would need to build MXNet by executing:
# docker build -f Dockerfile.build.ubuntu-17.04 -t mxnet.build.ubuntu-17.04 .
# if you haven't done it before.

FROM mxnet.build.ubuntu-17.04

ENV DEBIAN_FRONTEND=noninteractive

##################
# Julia installation
RUN wget -q https://julialang.s3.amazonaws.com/bin/linux/x64/0.5/julia-0.5.1-linux-x86_64.tar.gz\
  && tar -zxf julia-0.5.1-linux-x86_64.tar.gz\
  && rm julia-0.5.1-linux-x86_64.tar.gz\
  && ln -s $(pwd)/julia-6445c82d00/bin/julia /usr/bin/julia
##################


ENV MXNET_HOME /work/mxnet
WORKDIR /work/mxnet
RUN julia -e 'Pkg.add("MXNet")'




