# Developing MXNet

## Build MXNet from source with CMake

It's recommended that you install CMake and Ninja.

### Build and install google test

Google test needs to be compiled on each host once, it's required to compile and run the C++ unit tests.
You might have to adjust the source directory of libgtest (GTEST_SRCDIR below), in ubuntu 16.04 is /usr/src/gtest/ in
ubuntu 17.04 is /usr/src/googletest/googletest/

```bash
sudo apt-get install -y libgtest-dev
mkdir /tmp/gtest && cd /tmp/gtest
cmake $GTEST_SRCDIR -GNinja && ninja
sudo cp *.a /usr/local/lib
```

### Build MXNet

First we create a directory since CMake will do an out-of-source build. We choose the features and build flavour to use
by passing options to CMake. You can see the build options in the file `CMakeLists.txt`.

As an example we can do a minimal CPU debug build with OPENCV and OPENMP disabled. CMake will generate the build files.
The actual build is delegated to `ninja` in this case.

```bash
mkdir -p build && cd build
cmake -DUSE_CUDA=OFF -DUSE_OPENCV=OFF -DUSE_OPENMP=OFF -DCMAKE_BUILD_TYPE=Debug -GNinja ..  && ninja
```

You can check the available CMake options in CMakeLists.txt file: `cat CMakeLists.txt | grep mxnet_option`. Similar
options are needed if you want to use CLion. Under settings you can modify the CMake options so code navigation and
build works, for example if you are in a Mac laptop without cuda.

Then you can use the generated mxnet library `libxmnet.so` and install it with the python package with pip. It's
recommended that you use a python virtualenv or similar tool for having multiple installed versions and managing the
python interpreter.

```bash
virtualenv -p /usr/bin/python3 mxnet_py3
source mxnet_py3/bin/activate
```

Now we instal the python package with `-e` as editable so it's not copied to the python library location, and local
changes are reflected.

```bash
cp build/libmxnet.* ../python/mxnet
cd python/mxnet
pip3 install -e .
```

## Building in Ubuntu 17.04 with CUDA

Not all combinations of host C++ compiler and nvcc are supported

[cuda installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4lB5unFj4)

For Ubuntu 17.04 a solution is to explicitly set g++-4.9 and setting CCBIN as well:

```bash
sudo apt-get install -y g++-4.9 nvidia-cuda-dev nvidia-cuda-toolkit cmake ninja-build
CC=gcc-4.9 CXX=g++-4.9 CCBIN=g++-4.9 cmake -DUSE_CUDA=ON -DUSE_LAPACK=OFF\
 -DUSE_MKL_IF_AVAILABLE=OFF -DCUDA_VERBOSE_BUILD:BOOL=ON -GNinja ..
ninja -v
```

## Building in Mac

For building in mac you can install cmake, ninja and openblas with [homebrew](https://brew.sh/) and

```
mkdir build && cd build
cmake -DUSE_CUDA=OFF -DUSE_OPENCV=OFF -DUSE_OPENMP=OFF -DCMAKE_BUILD_TYPE=Debug -GNinja ..
ninja
```

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

Or the [mxnet slack channel](https://the-asf.slack.com/) to which you can ask for an invitation on the mail list.

