# -*- mode: dockerfile -*-
# part of the dockerfile to install the r binding

COPY install/r.sh install/
ADD https://raw.githubusercontent.com/dmlc/mxnet/master/R-package/DESCRIPTION  install/
RUN install/r.sh
RUN cd mxnet && make rpkg && R CMD INSTALL mxnet_current_r.tar.gz
