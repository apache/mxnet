#!/bin/bash
set -e

mkdir -p build/$PLATFORM$PLATFORM_EXTENSION/cmake
pushd build/$PLATFORM$PLATFORM_EXTENSION/cmake

if [[ -d "C:/msys64" ]] && [[ -z ${OpenBLAS_home:-} ]]; then
    export OpenBLAS_HOME=C:/msys64/mingw64/include/OpenBlas/
    export OpenBLAS=C:/msys64/mingw64/lib/
fi

CMAKE_FLAGS=
if [[ -d "${VCINSTALLDIR:-}" ]] && which ninja; then
    export CC="cl.exe"
    export CXX="cl.exe"
    CMAKE_FLAGS="-G Ninja -DCMAKE_CUDA_FLAGS=-Xcompiler=-bigobj"
fi

GPU_FLAGS="-DUSE_CUDA=OFF"
if [[ "$PLATFORM_EXTENSION" == *gpu ]]; then
    GPU_FLAGS="-DUSE_CUDA=ON -DMXNET_CUDA_ARCH=3.5"
fi

cmake $CMAKE_FLAGS -DCMAKE_BUILD_TYPE=Distribution -DCMAKE_INSTALL_PREFIX=$(pwd)/.. -DCMAKE_INSTALL_LIBDIR=lib $GPU_FLAGS -DUSE_OPENCV=OFF $(dirs +1)/..
cmake $(dirs +1)/..
cmake --build . --parallel $(getconf _NPROCESSORS_ONLN)
cmake --install .

popd
