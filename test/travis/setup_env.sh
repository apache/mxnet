#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "linux" ]; then
  export CXX="g++-4.8"
  export CC="gcc-4.8"
fi
