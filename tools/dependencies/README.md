# Overview

This folder contains scripts for building the dependencies from source. The static libraries from
the build artifacts can be used to create self-contained shared object for mxnet through static
linking.

# Settings

The scripts use the following environment variables for setting behavior:

`DEPS_PATH`: the location in which the libraries are downloaded, built, and installed.
`PLATFORM`: name of the OS in lower case. Supported options are 'linux' and 'darwin'.

It also expects the following build tools in path: make, cmake, tar, unzip, autoconf, nasm

# FAQ

## Build failure regarding to gcc, g++, gfortran
Currently, we only support gcc-4.8 build. It's your own choice to use a higher version of gcc. Please make sure your gcc, g++ and gfortran always have the same version in order to eliminate build failure.

## idn2 not found
This issue appeared in the OSX build with XCode version 8.0 above (reproduced on 9.2). Please add the following build flag in `curl.sh` if your XCode version is more than 8.0:
```
--without-libidn2
``` 