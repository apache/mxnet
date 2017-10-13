FROM ubuntu-17.04
RUN apt-get update &&\
  apt-get install -y python3-nose python-nose python-pip libgtest-dev valgrind ninja-build\
  &&\
  rm -rf /var/lib/apt/lists/*

###########################
# Unit tests
# Build google test
WORKDIR /work/googletest
RUN cmake /usr/src/googletest/googletest/ -GNinja
RUN ninja
# FIXME
RUN mkdir -p /usr/src/googletest/googletest/lib/
RUN cp libgtest.a /usr/src/googletest/googletest/lib/

ENV BUILD_OPTS "USE_OPENCV=0 USE_BLAS=openblas GTEST_PATH=/usr/src/googletest/googletest"

WORKDIR /work/mxnet
RUN make -j$(nproc) test $BUILD_OPTS
ENV MXNET_ENGINE_INFO=true
RUN build/tests/cpp/mxnet_test
RUN valgrind build/tests/cpp/mxnet_test
############################

############################
# Python tests
WORKDIR /work/mxnet/python
RUN pip3 install -e .
RUN pip install -e .

WORKDIR /work/mxnet
RUN nosetests3 --verbose tests/python/unittest
RUN nosetests --verbose tests/python/unittest
############################


############################
# Scala tests
RUN make scalatest $BUILD_OPTS
############################
