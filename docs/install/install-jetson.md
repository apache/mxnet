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

These instructions will walk through how to build MXNet and install the corresponding language bindings. Python is the default binding, but you may also try out one of the many language bindings MXNet has to offer. These instructions also cover how to setup MXNet's Java Inference API.

For the purposes of this install guide we will assume that CUDA is already installed on your Jetson device. The disk image provided by NVIDIA's getting started guides will have the Jetson toolkit preinstalled, and this also includes CUDA. You should double check what versions are installed and which version you plan to use.

You have several options for installing MXNet:
1. Use a Jetson MXNet pip wheel for Python development.
2. Use precompiled Jetson MXNet binaries.
3. Build MXNet from source
   * On a faster Linux computer using cross-compilation
   * On the Jetson itself (very slow and not recommended)


## Prerequisites
To build from source or to use the Python wheel, you must install the following dependencies on your Jetson.
Cross-compiling will require dependencies installed on that machine as well.

### Python API

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

<!--
### Java API

To use the Java inference API you only need the following dependencies:

```
sudo apt install \
                        maven \
                        openjdk-8-java
```

You may try to build the Java API .jar file yourself. If so, you will need these dependencies on that computer as well.

**Note:** The `mvn install` option for building the Java API .jar after compiling the MXNet binary files is only available on MXNet >= v1.5.0.
-->

### Configure CUDA

You can check to see what version of CUDA is running with `nvcc`.

```bash
nvcc --version
```

To switch CUDA versions on a device or computer that has more than one version installed, use the following and replace the version as appropriate.

```bash
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda
```

**Note:** When cross-compiling, change the CUDA version on the host computer you're using to match the version you're running on your Jetson device.

### Download the source & setup some environment variables:

These steps are optional, but some of the following instructions expect MXNet source files and the `MXNET_HOME` environment variable.

Clone the MXNet source code repository using the following `git` command in your home directory:

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet
```

Setup your environment variables for MXNet.

```bash
cd ..
export MXNET_HOME=$(pwd)
echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.rc
source ~/.rc
```

**Note:** Change the `~/.rc` steps according to how you prefer to use your shell. Otherwise, your environment variables will be gone after you logout.


## Install MXNet for Python

To use a prepared Python wheel, download it to your Jetson, and run it.
The following wheel was cross-compiled for Jetson using MXNet v1.4.1.
* https://s3.us-east-2.amazonaws.com/mxnet-public/install/jetson/1.4.1/mxnet-1.4.1-py3-none-any.whl

It should download the required dependencies, but if you have issues,
install the dependencies in the prerequisites section, then run the pip wheel.

```bash
sudo pip install mxnet-1.4.1-py3-none-any.whl
```

<!--
## Install MXNet for Java

The MXNet Java API for Jetson devices is still experimental. For best results try with the master branch.

### Use a pre-built jar

The package for Jetson is not currently on Maven, but you can download it from S3 instead.
The following jar was cross-compiled for Jetson using MXNet v1.4.1.
* https://s3.us-east-2.amazonaws.com/mxnet-public/install/jetson/1.4.1/mxnet-full_2.11-INTERNAL.jar

Place the file on your Jetson where your project can find it.

### Build your own jar

You can build a jar from any pre-compiled MXNet binary.
Follow the build from source instructions first. Once you have build the `libmxnet.so` file you are ready to build your jar files.

```bash
cd $MXNET_HOME/scala-package
mvn package
```

This will generate a file named similar to `src/lib/mxnet-full_2.11-INTERNAL.jar`.

### Use the MXNet-Java jar in your project

MXNet-Java can be easily included in your Maven managed project.
The following is an example entry for your project's `.pom` file.

```
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-INTERNAL</artifactId>
  <version>1.4.1</version>
  <systemPath>${basedir}/src/lib/mxnet-full_2.11-INTERNAL.jar</systemPath>
</dependency>
```

Refer to the [Java setup](https://mxnet.incubator.apache.org/versions/master/install/java_setup.html) page for further information.
-->

## Use a Pre-compiled MXNet Binary

If you want to just use the pre-compiled binary you can download it from S3:
* https://s3.us-east-2.amazonaws.com/mxnet-public/install/jetson/1.4.1/libmxnet.so

Place this file in `$MXNET_HOME/lib`.

To use this with the MXNet Python binding, you must match the source directory's checked out version with the binary's source version, then install it with pip.

```bash
cd $MXNET_HOME
git checkout v1.4.x
git submodule update --init
cd python
sudo pip install -e .
```

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

### Manual

**Step 1** Build the Shared Library

(Skip this sub-step for compiling on the Jetson device directly.)
Edit the Makefile to install the MXNet with CUDA bindings to leverage the GPU on the Jetson:

```bash
cp $MXNET_HOME/make/crosscompile.jetson.mk config.mk
```

Edit the Mshadow Makefile to ensure MXNet builds with Pascal's hardware level low precision acceleration by editing `3rdparty/mshadow/make/mshadow.mk`.
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
