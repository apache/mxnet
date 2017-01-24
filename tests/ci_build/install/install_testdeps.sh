#!/usr/bin/env bash

set -e

pip install cpplint 'pylint==1.4.4' 'astroid==1.3.6'

git clone https://github.com/google/googletest.git
cd googletest/googletest/make
make PREFIX=/usr/local
cd ..
export GTEST_DIR=${PWD}
g++ -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc
ar -rv libgtest.a gtest-all.o
mkdir -p /usr/local/gtest/include
mkdir -p /usr/local/gtest/lib
cp libgtest.a /usr/local/gtest/lib
cp -r include/ /usr/local/gtest/
export LD_LIBRARY_PATH=/usr/local/gtest/lib:$LD_LIBRARY_PATH

pip3 install nose
ln -s -f /opt/bin/nosetests /usr/local/bin/nosetests3
ln -s -f /opt/bin/nosetests-3.4 /usr/local/bin/nosetests-3.4
