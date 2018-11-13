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



