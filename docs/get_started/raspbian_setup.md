# Installing MXNet on Ubuntu
MXNet currently supports the Raspbian operating system for Raspberry Pi Devices, offering a python API. We currently reccomend running MxNet on the Raspberry Pi 3 or other devices with more than 1GB of RAM and with an SD card that has at least 4 GB free.
The Raspberry Pi 1, 2, Zero and other devices with less than 1GB of RAM are not sufficient to run the full MxNet library (though they can run the mxnet amalgamation library). 

## Standard installation

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.


### Build the Shared Library

On Rasbian versions Wheezy and later, you need the following dependencies:

- Git (to pull code from GitHub)

- libblas (for linear algebraic operations)

- libopencv (for computer vision operations this is optional if you want to save RAM and Disk Space)

Install these dependencies using the following commands:

```bash
    sudo apt-get update
    sudo apt-get -y install git cmake build-essential g++-4.8 c++-4.8 liblapack* libblas* libopencv*
```

Clone the MXNet source code repository to your computer, using ```git```.
```bash
    git clone https://github.com/dmlc/mxnet.git --recursive
```

Then build the full MxNet library.
```bash
    cd mxnet
    export ARM=1
    make
```

If you are not processing images with MxNet on the Pi you can install this package without opencv to minimize the size of the loaded library. You can do this by compiling with the following code:
```bash
    cd mxnet
    export ARM=1
    export USE_OPENCV = 0
    make
```

Executing either of these commands creates a library called ```libmxnet.so```.

### Install MXNet for Python On Raspberry Pi

To install python bindings run the following commands:

```bash
cd setup-utils
bash ./install-mxnet-ubuntu-python.sh
```

Sometimes this you have to run the bash command twice as it may fail the first time.

You are now ready to run MxNet on your Raspberry Pi! 

**Note - ** Keep in mind loading the entire MxNet library takes up a significant fraction of RAM, so steps such as switching off the GUI to free RAM may be nessecary to load training data or large models into memory.


## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
