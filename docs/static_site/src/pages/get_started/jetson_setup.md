---
layout: page
title: Jetson Setup
action: Get Started
action_url: /get_started
permalink: /get_started/jetson_setup
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


# Install MXNet on a Jetson

MXNet supports the Ubuntu Arch64 based operating system so you can run MXNet on NVIDIA Jetson Devices, such as the [TX2](http://www.nvidia.com/object/embedded-systems-dev-kits-modules.html) or [Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).

These instructions will walk through how to build MXNet and install MXNet's Python language binding.

For the purposes of this install guide we will assume that CUDA is already installed on your Jetson device. The disk image provided by NVIDIA's getting started guides will have the Jetson toolkit preinstalled, and this also includes CUDA. You should double check what versions are installed and which version you plan to use.

After installing the prerequisites, you have several options for installing MXNet:
1. Use a Jetson MXNet pip wheel for Python development and use a precompiled Jetson MXNet binary.
3. Build MXNet from source
   * On a faster Linux computer using cross-compilation
   * On the Jetson itself (very slow and not recommended)


## Prerequisites
To build from source or to use the Python wheel, you must install the following dependencies on your Jetson.
Cross-compiling will require dependencies installed on that machine as well.

### Python Dependencies

To use the Python API you need the following dependencies:

```bash
sudo apt update
sudo apt -y install \
                        build-essential \
                        git \
                        graphviz \
                        libatlas-base-dev \
                        libopencv-dev \
                        python-pip

sudo pip install --upgrade \
                        pip \
                        setuptools

sudo pip install \
                        graphviz==0.8.4 \
                        jupyter \
                        numpy==1.15.2
```

If you plan to cross-compile you will need to install these dependencies on that computer as well.
If you get an error about something being busy, you can restart the Nano and this error will go away. You can then continue installation of the prerequisites.

### Download the source & setup some environment variables:

These steps are optional, but some of the following instructions expect MXNet source files and the `MXNET_HOME` environment variable. Also, CUDA commands will not work out of the box without updating your path.

Clone the MXNet source code repository using the following `git` command in your home directory:

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
```

Setup your environment variables for MXNet and CUDA in your `.profile` file in your home directory.
Add the following to the file.

```bash
export PATH=/usr/local/cuda/bin:$PATH
export MXNET_HOME=$HOME/mxnet/
export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH
```

You can then apply this change immediately with the following:
```bash
source .profile
```

**Note:** Change the `~/.profile` steps according to how you prefer to use your shell. Otherwise, your environment variables will be gone after you logout.

### Configure CUDA

You can check to see what version of CUDA is running with `nvcc`.

```bash
nvcc --version
```

To switch CUDA versions on a device or computer that has more than one version installed, use the following and replace the symbolic link to the version you want. This one uses CUDA 10.0, which is preinstalled on the Nano.

```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda
```

**Note:** When cross-compiling, change the CUDA version on the host computer you're using to match the version you're running on your Jetson device.
**Note:** CUDA 10.1 is recommended but doesn't ship with the Nano's SD card image. You may want to go through CUDA upgrade steps first.


## Option 1. Install MXNet for Python

To use a prepared Python wheel, download it to your Jetson, and run it.
* [MXNet 1.4.0 - Python 3](https://s3.us-east-2.amazonaws.com/mxnet-public/install/jetson/1.4.0/mxnet-1.4.0-cp36-cp36m-linux_aarch64.whl)
* [MXNet 1.4.0 - Python 2](https://s3.us-east-2.amazonaws.com/mxnet-public/install/jetson/1.4.0/mxnet-1.4.0-cp27-cp27mu-linux_aarch64.whl)


It should download the required dependencies, but if you have issues,
install the dependencies in the prerequisites section, then run the pip wheel.

```bash
sudo pip install mxnet-1.4.0-cp27-cp27mu-linux_aarch64.whl
```

Now use a pre-compiled binary you can download it from S3 which is a patch v1.4.1:
* https://s3.us-east-2.amazonaws.com/mxnet-public/install/jetson/1.4.1/libmxnet.so

Place this file in `$MXNET_HOME/lib`.

To use this with the MXNet Python binding, you must match the source directory's checked out version with the binary's source version, then install it with pip.

```bash
cd $MXNET_HOME
git checkout v1.4.x
git submodule update --init --recursive
cd python
sudo pip install -e .
```

Refer to the following Conclusion and Next Steps section to test your installation.

## Option 2. Build MXNet from Source

Installing MXNet from source is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

You can use a Docker method or you can build from source manually.

### Docker

You must have installed Docker and be able to run `docker` without `sudo`.
Follow these [setup instructions to get to this point](https://docs.docker.com/install/linux/#manage-docker-as-a-non-root-user).
Then run the following to execute cross-compilation via Docker.

```bash
$MXNET_HOME/ci/build.py -p jetson
```

### Manual

**Step 1** Build the Shared Library

(Skip this sub-step for compiling on the Jetson device directly.)
Edit the Makefile to install the MXNet with CUDA bindings to leverage the GPU on the Jetson:

```bash
cp $MXNET_HOME/make/crosscompile.jetson.mk config.mk
```

Now edit `config.mk` to make some additional changes for the Nano. Update the following settings:

1. Update the CUDA path. `USE_CUDA_PATH = /usr/local/cuda`
2. Add `-gencode arch=compute-63, code=sm_62` to the `CUDA_ARCH` setting.
3. Update the NVCC settings. `NVCCFLAGS := -m64`
4. (optional, but recommended) Turn on OpenCV. `USE_OPENCV = 1`

Now edit the Mshadow Makefile to ensure MXNet builds with Pascal's hardware level low precision acceleration by editing `3rdparty/mshadow/make/mshadow.mk`.
The last line has `MSHADOW_USE_PASCAL` set to `0`. Change this to `1` to enable it.

```bash
MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1
```

Now you can build the complete MXNet library with the following command:

```bash
cd $MXNET_HOME
make -j $(nproc)
```

Executing this command creates a file called `libmxnet.so` in the `mxnet/lib` directory.

**Step 2** Install MXNet Python Bindings (optional)

To install Python bindings run the following commands in the MXNet directory:

```bash
cd $MXNET_HOME/python
sudo pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

**Step 3** Install the MXNet Java & Scala Bindings (optional)

Change directories to `scala-package` and run `mvn install`.

```bash
cd $MXNET_HOME/scala-package
mvn install
```

This creates the required `.jar` file to use in your Java or Scala projects.

## Conclusion and Next Steps

You are now ready to run MXNet on your NVIDIA Jetson TX2 or Nano device.
You can verify your MXNet Python installation with the following:

```python
import mxnet
mxnet.__version__
```

You can also verify MXNet can use your GPU with the following test:

```python
import mxnet as mx
a = mx.nd.ones((2, 3), mx.gpu())
b = a * 2 + 1
b.asnumpy()
```

If everything is working, it will report the version number.
For assistance, head over to the [MXNet Forum](https://discuss.mxnet.io/).
