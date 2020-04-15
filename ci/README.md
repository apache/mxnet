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

# Containerized build & test utilities

This folder contains scripts and dockerfiles used to build and test MXNet using
Docker containers

You need docker and nvidia docker if you have a GPU.

Also you need to run `pip3 install docker` as it uses the [docker python module](https://docker-py.readthedocs.io/en/stable/containers.html#)

If you are in ubuntu an easy way to install Docker CE is executing the
following script:


```
#!/bin/bash
set -e
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get -y install curl
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
         stable"
apt-get update
apt-get -y install docker-ce
service docker restart
usermod -a -G docker $SUDO_USER
```

For detailed instructions go to the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).


## build.py

The main utility to build is build.py which will run docker and mount the mxnet
folder as a volume to do in-place builds.

The build.py script does two functions, build the docker image, and it can be
also used to run commands inside this image with the propper mounts and
paraphernalia required to build mxnet inside docker from the sources on the
parent folder.

A set of helper shell functions are in `docker/runtime_functions.sh`.
`build.py` without arguments or `build.py --help` will display usage
information about the tool.

To build for armv7 for example:

```
./build.py -p armv7
```


To work inside a container with a shell you can do:

```
./build.py -p ubuntu_cpu -i
```

When building, the artifacts are located in the build/ directory in the project root. In case
`build.py -a` is invoked, the artifacts are located in build.<platform>/

# Docker container cleanup (Zombie containers)
Docker has a client-server architecture, so when the program that is executing the docker client
dies or receieves a signal, the container keeps running as it's started by the docker daemon.
We implement signal handlers that catch sigterm and sigint and cleanup containers before exit. In
Jenkins there's not enough time between sigterm and sigkill so we guarantee that containers are not
left running by propagating environment variables used by the Jenkins process tree killer to
identify which process to kill when the job is stopped. This has the effect of stopping the
container given that the process inside the container is terminated.

How to test this is working propperly: On the console you can hit ^C while a container is running
(not just building) and see that the container is stopped by running `docker ps` on another
terminal. In Jenkins this has been tested by stopping the job which has containers running and
verifying that the container stops shortly afterwards by running docker ps.

## Add a platform

To add a platform, you should add the appropriate dockerfile in
docker/Dockerfile.build.<platform> and add a shell function named
build_<platform> to the file docker/runtime_functions.sh with build
instructions for that platform.

## Warning
Due to current limitations of the CMake build system creating artifacts in the
source 3rdparty folder of the parent mxnet sources concurrent builds of
different platforms is NOT SUPPORTED.

## ccache
For all builds a directory from the host system is mapped where ccache will store cached
compiled object files (defaults to /tmp/ci_ccache). This will speed up rebuilds
significantly. You can set this directory explicitly by setting CCACHE_DIR environment
variable. All ccache instances are currently set to be 10 Gigabytes max in size.


## Testing with ARM / Edge devices with QEMU

We build on [QEMU](https://www.qemu.org/) and Linux [Kernel Support for
miscellaneous Binary
Formats](https://www.kernel.org/doc/html/v5.6/admin-guide/binfmt-misc.html) for
testing MXNet on edge devices. Test can be invoked with the same syntax as for
non-virtualized platforms:

```
./build.py -p armv7
./build.py -p test.armv7 /work/runtime_functions.sh unittest_ubuntu_python3_armv7
```

For the test step to succeed, you must run Linux kernel 4.8 or later and have qemu installed.

On Debian and Ubuntu systems, run the following command to install the dependencies:
```
sudo apt install binfmt-support qemu-user-static

# Use qemu-binfmt-conf.sh to register all binary types with the kernel
wget https://raw.githubusercontent.com/qemu/qemu/stable-4.1/scripts/qemu-binfmt-conf.sh
chmod +x qemu-binfmt-conf.sh
sudo ./qemu-binfmt-conf.sh --persistent yes --qemu-suffix "-static" --qemu-path "/usr/bin" --systemd ALL
```

If you run into segmentation faults at the beginning of the emulated tests, you
probably have a ancient version of Qemu on your system (or found a bug in
upstream Qemu). In that situation, you can rely on the
`multiarch/qemu-user-static` Docker project to register a set of up-to-date Qemu
binaries from their Docker image with your kernel:

```
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```
