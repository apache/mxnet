#!/bin/bash

echo "##########################"
echo $TRAVIS_OS_NAME

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
  brew update >/dev/null 2>&1
  brew tap homebrew/science
  brew info opencv
  brew install graphviz
  brew install opencv
fi

if [ ${TRAVIS_OS_NAME} == "linux" ]; then
  mkdir shadow_bin
  ln -s `which gcc-4.8` shadow_bin/gcc
  ln -s `which g++-4.8` shadow_bin/g++

  export PATH=$PWD/shadow_bin:$PATH
fi
