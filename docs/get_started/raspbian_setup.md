# Installing MXNet on Raspbian
MXNet supports the Debian based Raspbian ARM based operating system so you can run MXNet on Raspberry Pi Devices.

These instructions will walk through how to build MXNet for the Raspberry Pi and install the Python bindings for the library.

The complete MXNet library and its requirements can take almost 200MB of RAM, and loading large models with the library can take over 1GB of RAM. Because of this, we recommend running MXNet on the Raspberry Pi 3 or an equivalent device that has more than 1 GB of RAM and a Secure Digital (SD) card that has at least 4 GB of free memory.

## Installing MXNet

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

### Build the Shared Library

On Raspbian versions Wheezy and later, you need the following dependencies:

- Git (to pull code from GitHub)

- libblas (for linear algebraic operations)

- libopencv (for computer vision operations. This is optional if you want to save RAM and Disk Space)

- A C++ compiler that supports C++ 11. The C++ compiler compiles and builds MXNet source code. Supported compilers include the following:

- [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/)

Install these dependencies using the following commands in any directory:

```bash
    sudo apt-get update
    sudo apt-get -y install git cmake build-essential g++-4.8 c++-4.8 liblapack* libblas* libopencv*
```

Clone the MXNet source code repository using the following ```git``` command in your home directory:
```bash
    git clone https://github.com/dmlc/mxnet.git --recursive
    cd mxnet
```

If you aren't processing images with MXNet on the Raspberry Pi, you can minimize the size of the compiled library by building MXNet without the Open Source Computer Vision (OpenCV) library with the following commands:
```bash
    export USE_OPENCV = 0
    make
```

Otherwise, you can build the complete MXNet library with the following command: 
```bash
    make
```

Executing either of these commands start the build process, which can take up to a couple hours, and creates a file called ```libmxnet.so``` in the mxnet/lib directory.

If you are getting build errors in which the compiler is being killed, it is likely that the compiler is running out of memory (espeically if you are on Raspberry Pi 1, 2 or Zero, which have less than 1GB of RAM), this can often be rectified by increasing the swapfile size on the Pi by editing the file /etc/dphys-swapfile and changing the line CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024, then running:
```bash
	sudo /etc/init.d/dphys-swapfile stop
	sudo /etc/init.d/dphys-swapfile start
	free -m # to verify the swapfile size has been increased
```

## Install MXNet Python Bindings

To install python bindings run the following commands in the MXNet directory:

```bash
    cd python
    sudo python setup.py install
```

You are now ready to run MXNet on your Raspberry Pi device.

*Note - Because the complete MXNet library takes up a significant amount of the Raspberry Pi's limited RAM, when loading training data or large models into memory, you might have to turn off the GUI and terminate running processes to free RAM.*

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html#embedded)
