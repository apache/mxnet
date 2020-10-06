#!/bin/bash
set -e

mkdir -p build/$PLATFORM/cmake
pushd build/$PLATFORM/cmake

cmake -DCMAKE_BUILD_TYPE=Distribution -DCMAKE_INSTALL_PREFIX=$(pwd)/.. -DCMAKE_INSTALL_LIBDIR=lib -DUSE_CUDA=OFF -DUSE_OPENCV=OFF $(dirs +1)/..
cmake --build . --parallel $(getconf _NPROCESSORS_ONLN)
cmake --install .

popd
