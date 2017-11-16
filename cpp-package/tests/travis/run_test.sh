#!/bin/bash

if [ ${TASK} == "lint" ]; then
    make lint || exit -1
    echo "Check documentations of c++ code..."
    make doc 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
    echo "---------Error Log----------"
    cat logclean.txt
    echo "----------------------------"
    (cat logclean.txt|grep warning) && exit -1
    (cat logclean.txt|grep error) && exit -1
    exit 0
fi

if [ ${TRAVIS_OS_NAME} == "linux" ]; then
  # use g++-4.8 in linux
  export CXX=g++-4.8
fi

if [ ${TASK} == "build" ]; then
    make
    exit $?
fi
