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

MXNet supports Ubuntu AArch64 based operating system so you can run MXNet on all [NVIDIA Jetson modules](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/), such as Jetson Nano, TX1, TX2, Xavier NX and AGX Xavier.

These instructions will walk through how to build MXNet and install MXNet's Python language binding.

For the purposes of this install guide we will assume that CUDA is already installed on your Jetson device. NVIDIA Jetpack comes with the latest OS image for Jetson mdoule, and developer tools for both host computer and developer kit, and this also includes CUDA. You should double check what versions are installed and which version you plan to use.

After installing the prerequisites, you have several options for installing MXNet:
1. Build MXNet from source
   * On a faster Linux computer using cross-compilation
   * On the Jetson itself (very slow and not recommended)
2. Use a Jetson MXNet pip wheel for Python development and use a precompiled Jetson MXNet binary (not provided on this page as CUDA enabled wheels are not in accordance with ASF policy, users can download them from other 3rd party sources)

## Prerequisites
To build from source or to use the Python wheel, you must install the following dependencies on your Jetson.
Cross-compiling will require dependencies installed on that machine as well.

### Python Dependencies

To use the Python API you need the following dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
                        build-essential \
                        git \
                        libopenblas-dev \
                        libopencv-dev \
                        python3-pip \
                        python-numpy

sudo pip3 install --upgrade \
                        pip \
                        setuptools \
                        numpy
```

If you plan to cross-compile you will need to install these dependencies on that computer as well.

### Download the source & setup some environment variables:

These steps are optional, but some of the following instructions expect MXNet source files and the `MXNET_HOME` environment variable. Also, CUDA commands will not work out of the box without updating your path.

Clone the MXNet source code repository using the following `git` command in your home directory:

```bash
git clone --recursive https://github.com/apache/mxnet.git mxnet
```

You can also checkout a particular branch of MXNet. For example, to install MXNet v1.6:
```bash
git clone --recursive -b v1.6.x https://github.com/apache/mxnet.git mxnet
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

To switch CUDA versions on a device or computer that has more than one version installed, use the following and replace the symbolic link to the version you want. This one uses CUDA 10.2, which comes with Jetpack 4.4.

```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
```

**Note:** When cross-compiling, change the CUDA version on the host computer you're using to match the version you're running on your Jetson device.


## Build MXNet from Source

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

### Manually on the Jetson module (Slow)

**Step 1** Build the Shared Library

Use the config_jetson.mk Makefile to install MXNet with CUDA bindings to leverage the GPU on the Jetson module.

```bash
cp $MXNET_HOME/make/config_jetson.mk config.mk
```

The pre-existing Makefile builds for all Jetson architectures. Edit `config.mk` if you want to specifically build for a particular architecture or if you want to build without CUDA bindings (CPU only). You can make the following changes:

1. Modify `CUDA_ARCH` to build for specific architectures. Currently, we have `CUDA_ARCH = -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_72,code=sm_72`. Keep `-gencode arch=compute_53,code=sm_53` for Nano and TX1, `-gencode arch=compute_62,code=sm_62` for TX2, `-gencode arch=compute_72,code=sm_72` for Xavier NX and AGX Xavier.

2. For CPU only builds, remove `USE_CUDA_PATH`, `CUDA_ARCH`, `USE_CUDNN` flags.

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
sudo pip3 install -e .
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

You are now ready to run MXNet on your NVIDIA module.
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
