#!/bin/bash

echo "##########################"
echo $TRAVIS_OS_NAME

if [ ${TRAVIS_OS_NAME} == "linux" ]; then
  alias g++="g++-4.8"
  alias gcc="gcc-4.8"
fi

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
  brew update
  brew tap homebrew/science
  brew info opencv
  brew install graphviz
  brew install opencv
fi
