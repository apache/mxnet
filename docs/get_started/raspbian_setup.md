# Installing MXNet on Raspbian
MXNet currently supports a python API for the Debian based Raspbian operating system for Raspberry Pi Devices.

The full MxNet library is over 200MB when loaded into memory and the requirements can take almost 1GB of disk space. Due to the size we currently reccomend running MXNet on the Raspberry Pi 3 or equivilant devices with more than 1GB of RAM and with an SD card that has at least 4 GB of memory free. The Raspberry Pi 1, 2, Zero and other devices with less than 1GB of RAM are not sufficient to run the full MXNet library (though they can run the MXNet amalgamation library). 

## Standard installation

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

### Build the Shared Library

On Rasbian versions Wheezy and later, you need the following dependencies:

- Git (to pull code from GitHub)

- libblas (for linear algebraic operations)

- libopencv (for computer vision operations. This is optional if you want to save RAM and Disk Space)

- A C++ compiler that supports C++ 11. The C++ compiler compiles and builds MXNet source code. Supported compilers include the following:

- [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/)

Install these dependencies using the following commands:

```bash
    sudo apt-get update
    sudo apt-get -y install git cmake build-essential g++-4.8 c++-4.8 liblapack* libblas* libopencv*
```

Clone the MXNet source code repository to your computer, using ```git```.
```bash
    git clone https://github.com/dmlc/mxnet.git --recursive
```

If you are not processing images with MxNet on the Pi you can build MxNet without opencv to minimize the size of the compiled library. You can do this by compiling with the following code:
```bash
    cd mxnet
    export ARM=1
    export USE_OPENCV = 0
    make
```

Otherwise you can build the full MxNet library: 
```bash
    cd mxnet
    export ARM=1
    make
```

Whenever you are compiling MXNet for the Pi you need to ensure the "ARM" environment variable is set to 1 (this removes the x86 specific msse flag in the NNVM Submodule's Makefile, making it ARM compatible)

Executing either of these commands creates a library called ```libmxnet.so```.

## Install MXNet for Python On Raspberry Pi

To install python bindings run the following commands:

```bash
    cd python
    sudo python setup.py install
```

You are now ready to run MxNet on your Raspberry Pi! 

*Note - Keep in mind that loading the entire MXNet library takes up a significant fraction (200MB+) of the Raspberry Pi's limited RAM, so you might have to switch off the GUI and kill any concurrently running processes to free the RAM needed to load training data or large models into memory.*

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
