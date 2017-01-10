#!/bin/bash
#set -x
set -e

if ! brew --version > /dev/null 2>&1; then
  echo "Error: Brew is not installed or missing in your PATH"
  exit 1
fi

function brew_install() {
  echo "Checking for brew formula $1"
  brew ls --versions $1 || brew install $1
}
# You need to have brew installed, check: http://mxnet.io/get_started/osx_setup.html
#brew update
brew_install cmake
brew_install pkg-config
brew_install graphviz
brew_install openblas
brew tap homebrew/science
brew_install opencv
# For getting pip
brew_install python
# For visualization of network graphs
pip install graphviz
# Jupyter notebook
pip install jupyter


if [ ! -f config.mk ]; then
  cp -v make/osx.mk ./config.mk
else
  echo "config.mk already exists."
fi
echo "USE_BLAS = openblas" >> ./config.mk
echo "ADD_CFLAGS += -I/usr/local/opt/openblas/include" >> ./config.mk
echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib" >> ./config.mk
echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/" >> ./config.mk
make -j$(sysctl -n hw.ncpu)

echo "Installing the Python mxnet modules..."

cd python
sudo python setup.py install
cd ..

echo "All done."

