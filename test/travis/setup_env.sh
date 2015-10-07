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
