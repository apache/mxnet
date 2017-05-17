# Installing MXNet on The NVIDIA Jetson TX2
MXNet supports the Ubuntu Arch64 based operating system so you can run MXNet on NVIDIA Jetson Devices.

These instructions will walk through how to build MXNet for the Pascal based [NVIDIA Jetson TX2](http://www.nvidia.com/object/embedded-systems-dev-kits-modules.html) and install the corresponding python language bindings.

For the purposes of this install guide we will assume that CUDA is already installed on your Jetson device.

## Installing MXNet

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

### Build the Shared Library

You need the following additional dependencies:

- Git (to pull code from GitHub)

- libatlas (for linear algebraic operations)

- libopencv (for computer vision operations)

- python pip (to load relevant python packages for our language bindings)

Install these dependencies using the following commands in any directory:

```bash
    sudo apt-get update
    sudo apt-get -y install git build-essential libatlas-base-dev libopencv-dev graphviz python-pip 
    sudo pip install pip --upgrade
    sudo pip install setuptools numpy --upgrade
    sudo pip install graphviz jupyter
```

Clone the MXNet source code repository using the following ```git``` command in your home directory:
```bash
    git clone https://github.com/dmlc/mxnet.git --recursive
    cd mxnet
```

Edit the Makefile to install the MXNet with CUDA bindings to leverage the GPU on the Jetson:
```bash
    cp make/config.mk .
    echo "USE_CUDA=1" >> config.mk    
    echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
    echo "USE_CUDNN=1" >> config.mk
```

Edit the Mshadow Makefile to ensure MXNet builds with Pascal's hardware level low precision acceleration by editing mshadow/make/mshadow.mk and adding the following after line 122: 
```bash
MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1
```

Now you can build the complete MXNet library with the following command: 
```bash
    make -j $(nproc)
```

Executing this command creates a file called ```libmxnet.so``` in the mxnet/lib directory.

## Install MXNet Python Bindings

To install python bindings run the following commands in the MXNet directory:

```bash
    cd python
    sudo python setup.py install
    cd ..
    export MXNET_HOME=$(pwd)                       
    echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.bashrc
    source ~/.bashrc 
```

You are now ready to run MXNet on your NVIDIA Jetson TX2 device.
