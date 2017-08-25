# -*- mode: dockerfile -*-
# part of the dockerfile to install the scala binding

COPY install/scala.sh install/
RUN install/scala.sh

RUN cd mxnet && make scalapkg $BUILD_OPTS
