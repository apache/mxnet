# -*- mode: dockerfile -*-
# part of the dockerfile to install the julia binding

COPY install/julia.sh install/
RUN install/julia.sh
ENV MXNET_HOME /mxnet
RUN julia -e 'Pkg.add("MXNet")'
