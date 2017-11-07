# Developing MXNet

## Build MXNet from source with CMake

It's recommended that you install CMake and Ninja.

Chose the options that you want to compile with in a similar fashion as above. In particular this one is a debug CPU
build in OSx.

```
mkdir build && cd build
cmake -DUSE_CUDA=OFF -DUSE_OPENCV=OFF -DUSE_OPENMP=OFF -DCMAKE_BUILD_TYPE=Debug -GNinja ..
ninja
```
You can check the available CMake options in CMakeLists.txt file: `cat CMakeLists.txt | grep mxnet_option`. Similar
options are needed if you want to use CLion. Under settings you can modify the CMake options so code navigation and
build works.

Then you can use the library and install with pip. It's recommended that you use a python virtualenv or similar tool for having multiple
installed versions and managing the python interpreter.

```
virtualenv -p /usr/bin/python3.5 py3
source py3/bin/activate
```

```
cp build/libmxnet.* ../python/mxnet
cd python/mxnet
pip3 install -e .
```

## Building in Ubuntu 17.04 with CUDA

Not all combinations of host C++ compiler and nvcc are supported

[cuda installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4lB5unFj4)

For Ubuntu 17.04 a solution is to explicitly set g++-4.9 and setting CCBIN as well:

```
sudo apt-get install -y g++-4.9 nvidia-cuda-dev nvidia-cuda-toolkit cmake ninja-build
CC=gcc-4.9 CXX=g++-4.9 CCBIN=g++-4.9 cmake -DUSE_CUDA=ON -DUSE_LAPACK=OFF\
 -DUSE_MKL_IF_AVAILABLE=OFF -DCUDA_VERBOSE_BUILD:BOOL=ON -GNinja ..
ninja -v
```

## Building in Mac

For building in mac you can install cmake, ninja and openblas with [homebrew](https://brew.sh/) and
use one of the above cmake configurations.

## Further installation & build instructions:

See `docs/install/index.md`

## Running unit tests:

```
pip3 install nose
cd tests/python/unittest
nosetests -v . 2>&1 | tee test.log
```

For development questions you can subscribe to the [dev
list](https://lists.apache.org/list.html?dev@mxnet.apache.org)

Or the [mxnet slack channel](https://the-asf.slack.com/messages/C7FN4FCP9/)

