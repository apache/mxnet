# Development Tips

## Set up the Project
### Step 1. Obtain MXNet Library
The first step is to obtain the mxnet library. We recommend you build it from source. Also, you can download the library 
from 
#### Build from source
Refer to [Build From Source](https://mxnet.apache.org/get_started/build_from_source#building-mxnet)   
For MacOS users:
- Prepare  
```shell
# Install OS X Developer Tools
$ xcode-select --install

# Install Homebrew
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install dependencies
$ brew install cmake ninja ccache opencv
```
- Clone 3rd party projects
```shell
# Clone 3rd dependency for mxnet. It's necessary
$ git submodule update --init --recursive
```
- Build MXNet  
```shell
# select and copy cmake configure files for macos
$ cp config/darwin.cmake config.cmake

# create build directory for prpject
$ mkdir build; cd build

# cmake
$ cmake  ..
$ cmake --build .
```
Libraries will be generated under the directory _build/_.

For Linux users:  
Docker might help you build libraries on different platforms. You can get help from [README for CI](../ci/README.md).  
For example, you can build mxnet on Ubuntu with  by the following command.  
```shell
$ python3 ci/build.py -p ubuntu_cpu
```
##### Download Pre-built library
You can find the mxnet library from installed packages for mxnet, like python module. However, mxnet 2.0 is not released
yet, that's why we recommend you build it from source.  
```shell
# download python module for mxnet (have to mention that mxnet 2.0 hasn't been released by now)
$ pip3 install  mxnet==1.7.0.post2
# find the location of the installed module
$ python
Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46)
>>> import mxnet
>>> mxnet
<module 'mxnet' from '/Users/xxx/anaconda3/lib/python3.6/site-packages/mxnet/__init__.py'>
>>> quit()
# you can locate the module under /Users/xxx/anaconda3/lib/python3.6/site-packages/mxnet/
$ ls /Users/xxx/anaconda3/lib/python3.6/site-packages/mxnet/ | grep libmxnet
libmxnet.dylib
```
The compiled library is the file with the name of _libmxnet.*_. For MacOS, you will receive the file with suffix 
_.dylib_; For Linux, the lib file have the suffix ".so"; For Windows, the suffix is "." 

### Step 2. Build MXNet Native Lib for Java
The project uses gradle to manage dependencies. You can build the project using gradle. We have to encapsulate the mxnet
library into a jar file so that we can load it into JVM.
```shell
$ cd java-package
# Build the project
$ ./gradlew build 
# Create gradle tasks to package mxnet library into jar
# The task name is in this form {$favor}-{$platform}Jar
# MacOS -> mkl-osxJar
# Linux -> mkl-linuxJar
# Windows -> mkl-winJar
$ ./gradlew :native:buildLocalLibraryJarDefault
# Build native lib for macos
$ ./gradlew :mkl-osxJar
# Check the lib for osx
$ ls native/build/libs | grep osx
native-2.0.0-SNAPSHOT-osx-x86_64.jar
```
The jar file _native-2.0.0-SNAPSHOT-osx-x86_64.jar_ is the output lib file. 

### Step 3. Run Integration Test
When we execute the task for integration test, the built mxnet native lib will be added into classpath automatically. 
```shell
$ ./gradlew :integration:run

```