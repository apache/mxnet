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


## Testing with QEMU
To run the unit tests under qemu:
```
./build.py -p armv7 && ./build.py -p test.arm_qemu ./runtime_functions.py run_ut_py3_qemu
```

To get a shell on the container and debug issues with the emulator itself, we build the container
and then execute it interactively. We can afterwards use port 2222 on the host to connect with SSH.


```
ci/build.py -p test.arm_qemu -b && docker run -p2222:2222 -ti mxnetci/build.test.arm_qemu
```

Then from another terminal:

```
ssh -o StrictHostKeyChecking=no -p 2222 qemu@localhost
```

There are two pre-configured users: `root` and `qemu` both without passwords.


### Example of reproducing a test result with QEMU on ARM


You might want to enable a debug build first:

```
$ git diff
diff --git a/ci/docker/runtime_functions.sh b/ci/docker/runtime_functions.sh
index 39631f9..666ceea 100755
--- a/ci/docker/runtime_functions.sh
+++ b/ci/docker/runtime_functions.sh
@@ -172,6 +172,7 @@ build_armv7() {
         -DUSE_LAPACK=OFF \
         -DBUILD_CPP_EXAMPLES=OFF \
         -Dmxnet_LINKER_LIBS=-lgfortran \
+        -DCMAKE_BUILD_TYPE=Debug \
         -G Ninja /work/mxnet

     ninja -v

```

Then we build the project for armv7, the test container and start QEMU inside docker:

```
ci/build.py -p armv7
ci/build.py -p test.arm_qemu -b && docker run -p2222:2222 -ti mxnetci/build.test.arm_qemu
```



At this point we copy artifacts and sources to the VM, in another terminal (host) do the following:

```
# Copy mxnet sources to the VM
rsync --delete -e 'ssh -p2222' --exclude='.git/' -zvaP ./ qemu@localhost:mxnet


# Ssh into the vm
ssh -p2222 qemu@localhost

cd mxnet

# Execute a single failing C++ test
build/tests/mxnet_unit_tests --gtest_filter="ACTIVATION_PERF.ExecuteBidirectional"

# To install MXNet:
sudo pip3 install --upgrade --force-reinstall build/mxnet-1.3.1-py2.py3-none-any.whl

# Execute a single python test:

nosetests-3.4 -v -s tests/python/unittest/test_ndarray.py


# Debug with cgdb
sudo apt install -y libstdc++6-6-dbg
cgdb build/tests/mxnet_unit_tests

(gdb) !pwd
/home/qemu/mxnet
(gdb) set substitute-path /work /home/qemu
(gdb) set substitute-path /build/gcc-6-6mK9AW/gcc-6-6.3.0/build/arm-linux-gnueabihf/libstdc++-v3/include/ /usr/include/c++/6/
(gdb) r --gtest_filter="ACTIVATION_PERF.ExecuteBidirectional"
```
