# Installing MXNet

Indicate your preferred configuration. Then, follow the customized commands to install *MXNet*.

<script type="text/javascript" src='../../_static/js/options.js'></script>

<!-- START - OS Menu -->

<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Linux</button>
  <button type="button" class="btn btn-default opt">MacOS</button>
  <button type="button" class="btn btn-default opt">Cloud</button>
</div>
<br/>
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">Python</button>
</div>
<br/>
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active">CPU</button>
  <button type="button" class="btn btn-default opt">GPU</button>
</div>

<!-- Linux Python CPU options -->

<div class="linux">
  <div class="python">
    <div class="cpu gpu">
      <div class="btn-group opt-group" role="group">
        <button type="button" class="btn btn-default opt active">Pip</button>
        <button type="button" class="btn btn-default opt">Virtualenv</button>
        <button type="button" class="btn btn-default opt">Docker</button>
        <button type="button" class="btn btn-default opt">Build from Source</button>
      </div>
    </div>
  </div>
</div>

<!-- END - Main Menu -->

<!-- START - Linux Python CPU Installation Instructions -->

<div class="linux">
  <div class="python">
    <div class="cpu">

The following installation instructions have been tested on Ubuntu 14.04 and 16.04.

<div class="virtualenv">
<br/>

**Step 1**  Install virtualenv for Ubuntu.

```bash
$ sudo apt-get update
$ sudo apt-get install -y python-dev python-virtualenv
```

**Step 2**  Create and activate virtualenv environment for MXNet.

Following command creates a virtualenv environment at `~/mxnet` directory. However, you can choose any directory by replacing `~/mxnet` with a directory of your choice.

```bash
$ virtualenv --system-site-packages ~/mxnet
```

Activate the virtualenv environment created for *MXNet*.

```bash
$ source ~/mxnet/bin/activate
```

After activating the environment, you should see the prompt as below.

```bash
(mxnet)$
```

**Step 3**  Install MXNet in the active virtualenv environment.

Installing *MXNet* with pip requires a latest version of `pip`. Install the latest version of `pip` by issuing the following command.

```bash
(mxnet)$ pip install --upgrade pip
```

Install *MXNet* with OpenBLAS acceleration.

```bash
(mxnet)$ pip install mxnet
```

**Step 4**  Validate the installation by running simple *MXNet* code described [here](#validate-mxnet-installation).

**Note**  You can read more about virtualenv [here](https://virtualenv.pypa.io/en/stable/userguide/).

</div>

<div class="pip">
<br/>

**Step 1**  Install prerequisites - wget and latest pip.

Installing *MXNet* with pip requires a latest version of `pip`. Install the latest version of `pip` by issuing the following command in the terminal.

```bash
$ sudo apt-get update
$ sudo apt-get install -y wget python
$ wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
```

**Step 2** Install MXNet with OpenBLAS acceleration.

```bash
$ pip install mxnet
```

**Step 3**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div>

<div class="docker">
<br/>

Docker images with *MXNet* are available at [Docker Hub](https://hub.docker.com/r/mxnet/).

**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user) to allow managing docker containers without *sudo*.

If you skip this step, you need to use *sudo* each time you invoke Docker.

**Step 3** Pull the MXNet docker image.

```bash
$ docker pull mxnet/python # Use sudo if you skip Step 2
```

You can list docker images to see if mxnet/python docker image pull was successful.

```bash
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        latest              00d026968b3c        3 weeks ago         1.41 GB
```

**Step 4** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div>

<div class="build-from-source">
<br/>

Building *MXNet* from source is a 2 step process.
1. Build the *MXNet* core shared library, `libmxnet.so`, from the C++ sources.
2. Build the language specific bindings. Example - Python bindings, Scala bindings.

**Minimum Requirements**
1. [GCC 4.8](https://gcc.gnu.org/gcc-4.8/) or later to compile C++ 11.
2. [GNU Make](https://www.gnu.org/software/make/)

<br/>

**Build the MXNet core shared library**

**Step 1** Install build tools and git.
```bash
$ sudo apt-get update
$ sudo apt-get install -y build-essential git
```

**Step 2** Install OpenBLAS.

*MXNet* uses [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library for accelerated numerical computations on CPU machine. There are several flavors of BLAS libraries - [OpenBLAS](http://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/) and [MKL](https://software.intel.com/en-us/intel-mkl). In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
```bash
$ sudo apt-get install -y libopenblas-dev
```

**Step 3** Install OpenCV.

*MXNet* uses [OpenCV](http://opencv.org/) for efficient image loading and augmentation operations.
```bash
$ sudo apt-get install -y libopencv-dev
```

**Step 4** Download MXNet sources and build MXNet core shared library.

```bash
$ git clone --recursive https://github.com/dmlc/mxnet
$ cd mxnet
$ make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas
```

*Note* - USE_OPENCV and USE_BLAS are make file flags to set compilation options to use OpenCV and BLAS library. You can explore and use more compilation options in `make/config.mk`.

<br/>

**Build the MXNet Python binding**

**Step 1** Install prerequisites - python setup tools and numpy.

```bash
$ sudo apt-get install -y python-dev python-setuptools python-numpy
```

**Step 2** Build the MXNet Python binding.

```bash
$ cd python
$ sudo python setup.py install
```

**Step 3** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div>

</div>
</div>
</div>
<!-- END - Linux Python CPU Installation Instructions -->

<!-- START - Linux Python GPU Installation Instructions -->

<div class="linux">
<div class="python">
<div class="gpu">

The following installation instructions have been tested on Ubuntu 14.04 and 16.04.

**Prerequisites**

Install the following NVIDIA libraries to setup *MXNet* with GPU support:

1. Install CUDA 8.0 following the NVIDIA's [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
2. Install cuDNN 5 for CUDA 8.0 following the NVIDIA's [installation guide](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA for downloading the cuDNN library.

**Note:** Make sure to add CUDA install path to `LD_LIBRARY_PATH`.

Example - *export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH*

<div class="pip">
<br/>

**Step 1**  Install prerequisites - wget and latest pip.

Installing *MXNet* with pip requires a latest version of `pip`. Install the latest version of `pip` by issuing the following command in the terminal.

```bash
$ sudo apt-get update
$ sudo apt-get install -y wget python
$ wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
```

**Step 2**  Install *MXNet* with GPU support using CUDA 8.0

```bash
$ pip install mxnet-cu80
```

**Step 3**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div>

<div class="virtualenv">

<br/>

**Step 1**  Install virtualenv for Ubuntu.

```bash
$ sudo apt-get update
$ sudo apt-get install -y python-dev python-virtualenv
```

**Step 2**  Create and activate virtualenv environment for MXNet.

Following command creates a virtualenv environment at `~/mxnet` directory. However, you can choose any directory by replacing `~/mxnet` with a directory of your choice.

```bash
$ virtualenv --system-site-packages ~/mxnet
```

Activate the virtualenv environment created for *MXNet*.

```bash
$ source ~/mxnet/bin/activate
```

After activating the environment, you should see the prompt as below.

```bash
(mxnet)$
```

**Step 3**  Install MXNet in the active virtualenv environment.

Installing *MXNet* with pip requires a latest version of `pip`. Install the latest version of `pip` by issuing the following command.

```bash
(mxnet)$ pip install --upgrade pip
```

Install *MXNet* with GPU support using CUDA 8.0.

```bash
(mxnet)$ pip install mxnet-cu80
```

**Step 4**  Validate the installation by running simple *MXNet* code described [here](#validate-mxnet-installation).

**Note**  You can read more about virtualenv [here](https://virtualenv.pypa.io/en/stable/userguide/).

</div>

<div class="docker">

<br/>

Docker images with *MXNet* are available at [Docker Hub](https://hub.docker.com/r/mxnet/).

**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user) to allow managing docker containers without *sudo*.

If you skip this step, you need to use *sudo* each time you invoke Docker.

**Step 3** Install *nvidia-docker-plugin* following the [installation instructions](https://github.com/NVIDIA/nvidia-docker/wiki/Installation). *nvidia-docker-plugin* is required to enable the usage of GPUs from the docker containers.

**Step 4** Pull the MXNet docker image.

```bash
$ docker pull mxnet/python:gpu # Use sudo if you skip Step 2
```

You can list docker images to see if mxnet/python docker image pull was successful.

```bash
$ docker images # Use sudo if you skip Step 2

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        gpu                 493b2683c269        3 weeks ago         4.77 GB
```

**Step 5** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div>

<div class="build-from-source">

<br/>

Building *MXNet* from source is a 2 step process.
1. Build the *MXNet* core shared library, `libmxnet.so`, from the C++ sources.
2. Build the language specific bindings. Example - Python bindings, Scala bindings.

**Minimum Requirements**
1. [GCC 4.8](https://gcc.gnu.org/gcc-4.8/) or later to compile C++ 11.
2. [GNU Make](https://www.gnu.org/software/make/)

<br/>

**Build the MXNet core shared library**

**Step 1** Install build tools and git.
```bash
$ sudo apt-get update
$ sudo apt-get install -y build-essential git
```
**Step 2** Install OpenBLAS.

*MXNet* uses [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library for accelerated numerical computations. There are several flavors of BLAS libraries - [OpenBLAS](http://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/) and [MKL](https://software.intel.com/en-us/intel-mkl). In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
```bash
$ sudo apt-get install -y libopenblas-dev
```

**Step 3** Install OpenCV.

*MXNet* uses [OpenCV](http://opencv.org/) for efficient image loading and augmentation operations.
```bash
$ sudo apt-get install -y libopencv-dev
```

**Step 4** Download MXNet sources and build MXNet core shared library.

```bash
$ git clone --recursive https://github.com/dmlc/mxnet
$ cd mxnet
$ make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
```

*Note* - USE_OPENCV, USE_BLAS, USE_CUDA, USE_CUDA_PATH AND USE_CUDNN are make file flags to set compilation options to use OpenCV, OpenBLAS, CUDA and cuDNN libraries. You can explore and use more compilation options in `make/config.mk`. Make sure to set USE_CUDA_PATH to right CUDA installation path. In most cases it is - */usr/local/cuda*.

<br/>

**Build the MXNet Python binding**

**Step 1** Install prerequisites - python setup tools and numpy.

```bash
$ sudo apt-get install -y python-dev python-setuptools python-numpy
```

**Step 2** Build the MXNet Python binding.

```bash
$ cd python
$ sudo python setup.py install
```

**Step 3** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).
</div>

</div>
</div>
</div>
<!-- END - Linux Python GPU Installation Instructions -->

<div class="linux macos">

# Validate MXNet Installation

</div>

<div class="linux">
  <div class="python">
    <div class="cpu">

<div class="pip build-from-source">

Start the python terminal.

```bash
$ python
```
</div>

<div class="docker">

Launch a Docker container with `mxnet/python` image and run example *MXNet* python program on the terminal.

```bash
$ docker run -it mxnet/python bash # Use sudo if you skip Step 2 in the installation instruction

# Start a python terminal
root@4919c4f58cac:/# python
```
</div>

<div class="virtualenv">

Activate the virtualenv environment created for *MXNet*.

```bash
$ source ~/mxnet/bin/activate
```

After activating the environment, you should see the prompt as below.

```bash
(mxnet)$
```

Start the python terminal.

```bash
$ python
```

</div>

Run a short *MXNet* python program to create a 2X3 identity matrix, multiply each element in the matrix by 2 followed by adding 1. We expect output to be a 2X3 matrix with all elements being 3.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3))
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```
</div>
</div>
</div>

<!-- Validation for GPU machines -->

<div class="linux">
  <div class="python">
    <div class="gpu">

<div class="pip build-from-source">

Start the python terminal.

```bash
$ python
```
</div>

<div class="docker">

Launch a NVIDIA Docker container with `mxnet/python:gpu` image and run example *MXNet* python program on the terminal.

```bash
$ nvidia-docker run -it mxnet/python:gpu bash # Use sudo if you skip Step 2 in the installation instruction

# Start a python terminal
root@4919c4f58cac:/# python
```
</div>

<div class="virtualenv">

Activate the virtualenv environment created for *MXNet*.

```bash
$ source ~/mxnet/bin/activate
```

After activating the environment, you should see the prompt as below.

```bash
(mxnet)$
```

Start the python terminal.

```bash
$ python
```

</div>

Run a short *MXNet* python program to create a 2X3 identity matrix *a* on a *GPU*, multiply each element in the matrix by 2 followed by adding 1. We expect output to be a 2X3 matrix with all elements being 3. We use *mx.gpu()*, to set *MXNet* context to be GPUs.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```
</div>
</div>
</div>

<!-- Clean up -->
<div class="linux">
  <div class="python">
    <div class="cpu gpu">

<div class="pip build-from-source">

Exit the Python terminal.

```python
>>> exit()
$
```
</div>

<div class="virtualenv">

Exit the Python terminal and Deactivate the virtualenv *MXNet* environment.
```python
>>> exit()
(mxnet)$ deactivate
$
```

</div>

<div class="docker">

Exit the Python terminal and then the docker container.
```python
>>> exit()
root@4919c4f58cac:/# exit
```

</div>

</div>
</div>
</div>
