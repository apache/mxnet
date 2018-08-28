# Installing MXNet

Indicate your preferred configuration. Then, follow the customized commands to install *MXNet*.

<div class="dropdown">
  <button class="btn current-version btn-primary dropdown-toggle" type="button" data-toggle="dropdown">v1.2.1
  <span class="caret"></span></button>
  <ul class="dropdown-menu opt-group">
    <li class="opt active versions"><a href="#">v1.2.1</a></li>
    <li class="opt versions"><a href="#">v1.1.0</a></li>
    <li class="opt versions"><a href="#">v1.0.0</a></li>
    <li class="opt versions"><a href="#">v0.12.1</a></li>
    <li class="opt versions"><a href="#">v0.11.0</a></li>
    <li class="opt versions"><a href="#">master</a></li>
  </ul>
</div>

<script type="text/javascript" src='../_static/js/options.js'></script>

<!-- START - OS Menu -->

<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active platforms">Linux</button>
  <button type="button" class="btn btn-default opt platforms">MacOS</button>
  <button type="button" class="btn btn-default opt platforms">Windows</button>
  <button type="button" class="btn btn-default opt platforms">Cloud</button>
  <button type="button" class="btn btn-default opt platforms">Devices</button>
</div>

<!-- START - Language Menu -->

<div class="linux macos windows">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default opt active languages">Python</button>
  <button type="button" class="btn btn-default opt languages">Scala</button>
  <button type="button" class="btn btn-default opt languages">R</button>
  <button type="button" class="btn btn-default opt languages">Julia</button>
  <button type="button" class="btn btn-default opt languages">Perl</button>
  <button type="button" class="btn btn-default opt languages">Cpp</button>
</div>
</div>

<!-- No CPU GPU for other Devices -->
<div class="linux macos windows cloud">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default processors opt active">CPU</button>
  <button type="button" class="btn btn-default processors opt">GPU</button>
</div>
</div>

<!-- other devices -->
<div class="devices">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default iots opt active">Raspberry Pi</button>
  <button type="button" class="btn btn-default iots opt">NVIDIA Jetson</button>
</div>
</div>

<!-- Linux Python GPU Options -->

<div class="linux macos windows">
<div class="python">
<div class="cpu gpu">
<div class="btn-group opt-group" role="group">
  <button type="button" class="btn btn-default environs opt active">Pip</button>
  <button type="button" class="btn btn-default environs opt">Virtualenv</button>
  <button type="button" class="btn btn-default environs opt">Docker</button>
  <button type="button" class="btn btn-default environs opt">Build from Source</button>
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


<div class="pip">
<br/>

**Step 1**  Install prerequisites - wget and latest pip.

Installing *MXNet* with pip requires a latest version of `pip`. Install the latest version of `pip` by issuing the following command in the terminal.

```bash
$ sudo apt-get update
$ sudo apt-get install -y wget python gcc
$ wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
```

<div class="v1-2-1">

**Step 2** Install MXNet with OpenBLAS acceleration.

```bash
$ pip install mxnet
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-mkl
```

</div> <!-- End of v1-2-1 -->

<div class="v1-1-0">

**Step 2** Install MXNet with OpenBLAS acceleration.

```bash
$ pip install mxnet==1.1.0
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-mkl==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

**Step 2** Install MXNet with OpenBLAS acceleration.

```bash
$ pip install mxnet==1.0.0
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-mkl==1.0.0
```

</div> <!-- End of v1-0-0-->


<div class="v0-12-1">


**Step 2** Install MXNet with OpenBLAS acceleration.

```bash
$ pip install mxnet==0.12.1
```

For MXNet 0.12.0 -

```bash
$ pip install mxnet==0.12.0
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-mkl==0.12.1
```

For MXNet 0.12.0 -

```bash
$ pip install mxnet-mkl==0.12.0
```

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">


**Step 2** Install MXNet with OpenBLAS acceleration.

```bash
$ pip install mxnet==0.11.0
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-mkl==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">


**Step 2** Install MXNet with OpenBLAS acceleration.

```bash
$ pip install mxnet --pre
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-mkl --pre
```

</div> <!-- End of master-->

</div> <!-- End of pip -->

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
$ pip install --upgrade pip
```

<div class="v1-2-1">

Install *MXNet* with OpenBLAS acceleration.

```bash
$ pip install mxnet
```

</div> <!-- End of v1-2-1-->

<div class="v1-1-0">

Install *MXNet* with OpenBLAS acceleration.

```bash
$ pip install mxnet==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

Install *MXNet* with OpenBLAS acceleration.

```bash
$ pip install mxnet==1.0.0
```

</div> <!-- End of v1-0-0-->


<div class="v0-12-1">

Install *MXNet* with OpenBLAS acceleration.

```bash
$ pip install mxnet==0.12.1
```

For *MXNet* 0.12.0 -

```bash
$ pip install mxnet==0.12.0
```

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

Install *MXNet* with OpenBLAS acceleration.

```bash
$ pip install mxnet==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

Install *MXNet* with OpenBLAS acceleration.

```bash
$ pip install mxnet --pre
```

</div> <!-- End of master-->


**Step 4**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 5**  Validate the installation by running simple *MXNet* code described [here](#validate-mxnet-installation).

**Note**  You can read more about virtualenv [here](https://virtualenv.pypa.io/en/stable/userguide/).

</div> <!-- END of virtualenv -->


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

</div> <!-- END of docker -->

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

*MXNet* uses [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [LAPACK](https://en.wikipedia.org/wiki/LAPACK) libraries for accelerated numerical computations on CPU machine. There are several flavors of BLAS/LAPACK libraries - [OpenBLAS](http://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/) and [MKL](https://software.intel.com/en-us/intel-mkl). In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
```bash
$ sudo apt-get install -y libopenblas-dev liblapack-dev
```

**Step 3** Install OpenCV.

*MXNet* uses [OpenCV](http://opencv.org/) for efficient image loading and augmentation operations.
```bash
$ sudo apt-get install -y libopencv-dev
```

**Step 4** Download MXNet sources and build MXNet core shared library. You can clone the repository as described in the following code block, or you may try the <a href="download.html">download links</a> for your desired MXNet version.

```bash
$ git clone --recursive https://github.com/apache/incubator-mxnet
$ cd incubator-mxnet
$ make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas
```

*Note* - USE_OPENCV and USE_BLAS are make file flags to set compilation options to use OpenCV and BLAS library. You can explore and use more compilation options in `make/config.mk`.

<br/>

**Build the MXNet Python binding**

**Step 1** Install prerequisites - python, setup-tools, python-pip and libfortran (required for Numpy).

```bash
$ sudo apt-get install -y python-dev python-setuptools python-pip libgfortran3
```

**Step 2** Install the MXNet Python binding.

```bash
$ cd python
$ pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div><!-- END of build from source -->
</div><!-- END of CPU -->
<!-- END - Linux Python CPU Installation Instructions -->

<!-- START - Linux Python GPU Installation Instructions -->

<div class="gpu">

The following installation instructions have been tested on Ubuntu 14.04 and 16.04.


**Prerequisites**

Install the following NVIDIA libraries to setup *MXNet* with GPU support:

1. Install CUDA 9.0 following the NVIDIA's [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
2. Install cuDNN 7 for CUDA 9.0 following the NVIDIA's [installation guide](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA for downloading the cuDNN library.

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

<div class="v1-2-1">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.2

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

```bash
$ pip install mxnet-cu92
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-cu90mkl
```

</div> <!-- End of v1-2-1-->


<div class="v1-1-0">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.1

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

```bash
$ pip install mxnet-cu91==1.1.0
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install MXNet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-cu91mkl==1.1.0
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of v1-1-0-->


<div class="v1-0-0">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.0

```bash
$ pip install mxnet-cu90==1.0.0
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-cu90mkl==1.0.0
```

</div> <!-- End of v1-0-0-->


<div class="v0-12-1">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.0

```bash
$ pip install mxnet-cu90==0.12.1
```

For *MXNet* 0.12.0 -

```bash
$ pip install mxnet-cu90==0.12.0
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install mxnet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-cu90mkl==0.12.1
```

For *MXNet* 0.12.0 -

```bash
$ pip install mxnet-cu90mkl==0.12.0
```

</div> <!-- End of v0-12-1-->


<div class="v0-11-0">

**Step 2**  Install *MXNet* with GPU support using CUDA 8.0

```bash
$ pip install mxnet-cu80==0.11.0
```

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

**Experimental Choice** If You would like to install MXNet with Intel MKL, try the experimental pip package with MKL:
```bash
$ pip install mxnet-cu80mkl==0.11.0
```

</div> <!-- End of v0-11-0-->


</div> <!-- END of pip -->

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


<div class="v1-2-1">

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

Install *MXNet* with GPU support using CUDA 9.2:

```bash
(mxnet)$ pip install mxnet-cu92
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of v1-2-1-->


<div class="v1-1-0">

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

Install *MXNet* with GPU support using CUDA 9.1:

```bash
(mxnet)$ pip install mxnet-cu91==1.1.0
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of v1-1-0-->


<div class="v1-0-0">

Install *MXNet* with GPU support using CUDA 9.0.

```bash
(mxnet)$ pip install mxnet-cu90==1.0.0
```
Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of v1-0-0-->


<div class="v0-12-1">

Install *MXNet* with GPU support using CUDA 9.0.

```bash
(mxnet)$ pip install mxnet-cu90==0.12.1
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of v0-12-1-->


<div class="v0-11-0">

Install *MXNet* with GPU support using CUDA 8.0.

```bash
(mxnet)$ pip install mxnet-cu80==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

Install *MXNet* with GPU support using CUDA 9.2.

```bash
(mxnet)$ pip install mxnet-cu92 --pre
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of master-->

**Step 4**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 5**  Validate the installation by running simple *MXNet* code described [here](#validate-mxnet-installation).

**Note**  You can read more about virtualenv [here](https://virtualenv.pypa.io/en/stable/userguide/).

</div> <!-- END of virtualenv -->

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

</div> <!-- END of docker -->

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

*MXNet* uses [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [LAPACK](https://en.wikipedia.org/wiki/LAPACK) libraries for accelerated numerical computations on CPU machine. There are several flavors of BLAS/LAPACK libraries - [OpenBLAS](http://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/) and [MKL](https://software.intel.com/en-us/intel-mkl). In this step we install OpenBLAS. You can choose to install ATLAS or MKL.
```bash
$ sudo apt-get install -y libopenblas-dev liblapack-dev
```

**Step 3** Install OpenCV.

*MXNet* uses [OpenCV](http://opencv.org/) for efficient image loading and augmentation operations.
```bash
$ sudo apt-get install -y libopencv-dev
```

**Step 4** Download MXNet sources and build MXNet core shared library. You can clone the repository as described in the following code block, or you may try the <a href="download.html">download links</a> for your desired MXNet version.

```bash
$ git clone --recursive https://github.com/apache/incubator-mxnet
$ cd incubator-mxnet
$ make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
```

*Note* - USE_OPENCV, USE_BLAS, USE_CUDA, USE_CUDA_PATH AND USE_CUDNN are make file flags to set compilation options to use OpenCV, OpenBLAS, CUDA and cuDNN libraries. You can explore and use more compilation options in `make/config.mk`. Make sure to set USE_CUDA_PATH to right CUDA installation path. In most cases it is - */usr/local/cuda*.

<br/>

**Install the MXNet Python binding**

**Step 1** Install prerequisites - python, setup-tools, python-pip and libfortran (required for Numpy)..

```bash
$ sudo apt-get install -y python-dev python-setuptools python-pip libgfortran3
```

**Step 2** Install the MXNet Python binding.

```bash
$ cd python
$ pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Step 4** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div> <!-- END of build from source -->
</div> <!-- END of GPU -->
</div> <!-- END of Python -->
<!-- END - Linux Python GPU Installation Instructions -->


<div class="r">
<div class="cpu">

The default version of R that is installed with `apt-get` is insufficient. You will need to first [install R v3.4.4+ and build MXNet from source](ubuntu_setup.html#install-the-mxnet-package-for-r).

After you have setup R v3.4.4+ and MXNet, you can build and install the MXNet R bindings with the following, assuming that `incubator-mxnet` is the source directory you used to build MXNet as follows:

```bash
$ cd incubator-mxnet
$ make rpkg
```

</div> <!-- END of CPU -->


<div class="gpu">

The default version of R that is installed with `apt-get` is insufficient. You will need to first [install R v3.4.4+ and build MXNet from source](ubuntu_setup.html#install-the-mxnet-package-for-r).

After you have setup R v3.4.4+ and MXNet, you can build and install the MXNet R bindings with the following, assuming that `incubator-mxnet` is the source directory you used to build MXNet as follows:

```bash
$ cd incubator-mxnet
$ make rpkg
```

</div> <!-- END of GPU -->
</div> <!-- END of R -->


<div class="scala">
<div class="gpu">

You can use the Maven packages defined in the following `dependency` to include MXNet in your Scala project. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
</dependency>
```
<br>
</div> <!-- End of gpu -->

<div class="cpu">

You can use the Maven packages defined in the following `dependency` to include MXNet in your Scala project. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
</dependency>
```
<br>
</div> <!-- End of cpu -->
</div> <!-- End of scala -->


<div class="julia perl">
<div class="cpu gpu">

Follow the installation instructions [in this guide](./ubuntu_setup.md) to set up MXNet.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia perl -->


<div class="cpp">
<div class="cpu gpu">
<p> To build the C++ package, please refer to <a href="build_from_source.html#build-the-c-package">this guide</a>. </p>
<br/>
</div> <!-- End of cpu gpu -->
</div> <!-- END - C++-->
</div> <!-- END - Linux -->


<!-- START - MacOS Python CPU Installation Instructions -->

<div class="macos">
<div class="python">
<div class="cpu">

The following installation instructions have been tested on OSX Sierra and El Capitan.


<div class="pip">
<br/>

**Step 1**  Install prerequisites - Homebrew, python development tools.

```bash
# Install Homebrew
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ export PATH=/usr/local/bin:/usr/local/sbin:$PATH

# Install python development tools - python2.7, pip, python-setuptools
$ brew install python
```

**Step 2** Install MXNet with OpenBLAS acceleration.

Installing *MXNet* with pip requires a latest version of `pip`. Install the latest version of `pip` by issuing the following command.

```bash
$ pip install --upgrade pip
$ pip install --upgrade setuptools
```

<div class="v1-2-1">

Then use pip to install MXNet:

```bash
$ pip install mxnet
```
</div> <!-- End of v1-2-1 -->


<div class="v1-1-0">

Then use pip to install MXNet:

```bash
$ pip install mxnet==1.1.0
```

</div> <!-- End of v1-1-0-->


<div class="v1-0-0">

Then use pip to install MXNet:

```bash
$ pip install mxnet==1.0.0
```

</div> <!-- End of v1-0-0-->

<div class="v0-12-1">

Then use pip to install MXNet:

```bash
$ pip install mxnet=0.12.1
```

For MXNet 0.12.0 -

```bash
$ pip install mxnet=0.12.0
```


</div> <!-- End of v0-12-1-->


<div class="v0-11-0">

Then use pip to install MXNet:

```bash
$ pip install mxnet==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

Then use pip to install MXNet:

```bash
$ pip install mxnet --pre
```

</div> <!-- End of master-->

**Step 3**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
$ brew install graphviz
$ pip install graphviz
```

**Step 4**  Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div> <!-- END of pip -->


<div class="virtualenv">
<br/>

**Step 1**  Install prerequisites - Homebrew, python development tools.

```bash
# Install Homebrew
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ export PATH=/usr/local/bin:/usr/local/sbin:$PATH

# Install python development tools - python2.7, pip, python-setuptools
$ brew install python
```

**Step 2**  Install virtualenv for macOS.

```bash
$ pip install virtualenv
```

**Step 3**  Create and activate virtualenv environment for MXNet.

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

**Step 4**  Install MXNet in the active virtualenv environment.

Installing *MXNet* with pip requires a latest version of `pip`. Install the latest version of `pip` by issuing the following command.

```bash
(mxnet)$ pip install --upgrade pip
(mxnet)$ pip install --upgrade setuptools
```

<div class="v1-2-1">

Install *MXNet* with OpenBLAS acceleration.

```bash
(mxnet)$ pip install mxnet
```

</div> <!-- End of v1-2-1-->

<div class="v1-1-0">

Install *MXNet* with OpenBLAS acceleration.

```bash
(mxnet)$ pip install mxnet==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

Install *MXNet* with OpenBLAS acceleration.

```bash
(mxnet)$ pip install mxnet==1.0.0
```

</div> <!-- End of v1-0-0-->


<div class="v0-12-1">

Install *MXNet* with OpenBLAS acceleration.

```bash
(mxnet)$ pip install mxnet==0.12.1
```

For *MXNet* 0.12.0 -

```bash
(mxnet)$ pip install mxnet==0.12.0
```


</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

Install *MXNet* with OpenBLAS acceleration.

```bash
(mxnet)$ pip install mxnet==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

Install *MXNet* with OpenBLAS acceleration.

```bash
(mxnet)$ pip install mxnet --pre
```

</div> <!-- End of master-->


**Step 5**  Install [Graphviz](http://www.graphviz.org/). (Optional, needed for graph visualization using `mxnet.viz` package).
```bash
$ brew install graphviz
(mxnet)$ pip install graphviz
```

**Step 6**  Validate the installation by running simple *MXNet* code described [here](#validate-mxnet-installation).

**Note**  You can read more about virtualenv [here](https://virtualenv.pypa.io/en/stable/userguide/).

</div> <!-- End of virtualenv -->


<div class="docker">
<br/>

Docker images with *MXNet* are available at [Docker Hub](https://hub.docker.com/r/mxnet/).

**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/docker-for-mac/install/#install-and-run-docker-for-mac).

*Note* - You can install Community Edition (CE) to get started with *MXNet*.

**Step 2** Pull the MXNet docker image.

```bash
$ docker pull mxnet/python
```

You can list docker images to see if mxnet/python docker image pull was successful.

```bash
$ docker images

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mxnet/python        latest              00d026968b3c        3 weeks ago         1.41 GB
```

**Step 4** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div> <!-- END of docker -->


<div class="build-from-source">
<br/>

**Prerequisites**

If not already installed, [download and install Xcode](https://developer.apple.com/xcode/) (or [insall it from the App Store](https://itunes.apple.com/us/app/xcode/id497799835)) for macOS. [Xcode](https://en.wikipedia.org/wiki/Xcode) is an integrated development environment for macOS containing a suite of software development tools like C/C++ compilers, BLAS library and more.

<br/>

Building *MXNet* from source is a 2 step process.
1. Build the *MXNet* core shared library, `libmxnet.so`, from the C++ sources.
2. Build the language specific bindings. Example - Python bindings, Scala bindings.

Make sure you have installed Xcode before proceeding further.

<br/>

All the instructions to build *MXNet* core shared library and *MXNet* Python bindings are compiled as one helper *bash* script. You can use [this bash script](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-osx-python.sh) to build *MXNet* for Python, from source, on macOS.

**Step 1** Download the bash script for building MXNet from source.

```bash
$ curl -O https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-osx-python.sh
```

**Step 2** Run the script to get latest MXNet source and build.

```bash
# Make the script executable
$ chmod 744 install-mxnet-osx-python.sh

# Run the script. It takes around 5 mins.
$ bash install-mxnet-osx-python.sh
```

**Step 3** Validate the installation by running simple MXNet code described [here](#validate-mxnet-installation).

</div> <!-- END of build from source -->
</div> <!-- END of CPU -->


<!-- START - Mac OS Python GPU Installation Instructions -->
<div class="gpu">
<div class="pip virtualenv docker">
</br>

Try the **Build from Source** option for now.

</div>

<div class="build-from-source">

**Step 1**  Install prerequisites - Homebrew, python development tools.

```bash
# Install Homebrew
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ export PATH=/usr/local/bin:/usr/local/sbin:$PATH

# Install python development tools - python2.7, pip, python-setuptools
$ brew install python pkg-config graphviz
```

**Step 2**  Install optional components - OpenCV

If you want to use OpenCV you should install it first, then build MXNet with the `USE_OPENCV=1` option in the later steps.

```bash
brew tap homebrew/science
brew install opencv

```

**Step 3**  Install CUDA and cuDNN

The following instructions are for CUDA 9.1 and cuDNN 7 for macOS 10.12+ and a CUDA-capable GPU. They summarize confirmed successful builds in [#9217](https://github.com/apache/incubator-mxnet/issues/9217).
Alternatively, you may follow the [CUDA installation instructions for macOS](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html).

1. [Download Xcode 8.3.3 from Apple](https://developer.apple.com/download/more/). This is the version [NVIDIA specifies in its instructions for macOS](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html). Unzip and rename to `Xcode8.3.3.app`.

2. Run `sudo xcode-select -s /Applications/Xcode8.3.3.app` or to wherever you have placed Xcode.

3. Run `xcode-select --install` to install all command line tools, compilers, etc.

4. Run `sudo xcodebuild -license accept` to accept Xcode's licensing terms.

5. Install CUDA for macOS. Specific steps are provided in NVIDIA's [CUDA installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#installation).

6. [Download](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download-mac) and [install](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installmac) cuDNN for macOS. You will need to [create a free developer account](https://developer.nvidia.com/accelerated-computing-developer) with NVIDIA prior to getting the download link.

**Step 4**  Build MXNet

1. Run `git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet` to get the latest version.

2. Run `cd mxnet`.

3. Edit the `make/osx.mk` file to set the following parameters:

    ```
    USE_CUDA = 1
    USE_CUDA_PATH = /usr/local/cuda
    USE_CUDNN = 1
    USE_OPENCV = 0   # set to 1 if you want to build with OpenCV
    ```

4. Copy the `make/osx.mk` to `config.mk`

5. Run `make`. If you previously attempted to compile you might want to do `make clean_all` first. You can also run `make -j` with the number of processors you have to compile with multithreading. There'll be plenty of warnings, but there should be no errors.

6. Once finished, you should have a file called `libmxnet.so` in `lib/`.

7. Do `cd python`.

8. Run `sudo pip install -e .` **Note**: the `.` is part of the command.

</div> <!-- END of build from source -->
</div> <!-- END of GPU -->
</div> <!-- END of Python -->


<!-- START - MacOS R CPU Installation Instructions -->

<div class="r">
<div class="cpu">

Install the latest version (3.5.1+) of R from [CRAN](https://cran.r-project.org/bin/macosx/).
You can [build MXNet-R from source](osx_setup.html#install-the-mxnet-package-for-r), or you can use a pre-built binary:

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

</div> <!-- END of CPU -->


<div class="gpu">

Will be available soon.

</div> <!-- END of GPU -->
</div> <!-- END of R -->

<div class="scala">
<div class="cpu">

You can use the Maven packages defined in the following `dependency` to include MXNet in your Scala project. Please refer to the <a href="scala_setup.html">MXNet-Scala setup guide</a> for a detailed set of instructions to help you with the setup process.

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-mac cpu-green.svg" alt="maven badge"/></a>

```html
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
</dependency>
```
<br>
</div> <!-- End of cpu  -->
<div class="gpu">

Not available at this time. <br>

</div>
</div> <!-- End of scala -->


<div class="julia perl">
<div class="cpu gpu">

Follow the installation instructions [in this guide](./osx_setup.md) to set up MXNet.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia perl -->


<div class="cpp">
<p>To build the C++ package, please refer to <a href="build_from_source.html#build-the-c-package">this guide</a>.</p>
<br/>
</div>
</div> <!-- END - Mac OS -->









<div class="windows">
<div class="python">
<div class="cpu">
<div class="pip">

<br/>

**Step 1**  Install Python.

[Anaconda](https://www.anaconda.com/download/) is recommended.

<div class="v1-2-1">

**Step 2**  Install *MXNet*.

```bash
$ pip install mxnet
```

</div> <!-- End of v1-2-1-->

<div class="v1-1-0">

**Step 2**  Install *MXNet*.

```bash
$ pip install mxnet==1.1.0
```

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

**Step 2**  Install *MXNet*.

```bash
$ pip install mxnet==1.0.0
```

</div> <!-- End of v1-0-0-->


<div class="v0-12-1">

**Step 2**  Install *MXNet*.

```bash
$ pip install mxnet==0.12.1
```

For *MXNet* 0.12.0 -

```bash
$ pip install mxnet==0.12.0
```


</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

**Step 2**  Install *MXNet*.

```bash
$ pip install mxnet==0.11.0
```


</div> <!-- End of v0-11-0-->

<div class="master">

**Step 2**  Install *MXNet*.

```bash
$ pip install mxnet --pre
```

</div> <!-- End of master-->


</div> <!-- End of pip -->


<div class="virtualenv docker build-from-source">

Follow the installation instructions [in this guide](./windows_setup.md) to set up MXNet.

</div> <!-- End of virtualenv docker build-from-source -->
</div> <!-- End of CPU -->


<div class="gpu">
<div class="pip">

<br/>

**Step 1**  Install Python.

[Anaconda](https://www.anaconda.com/download/) is recommended.


<div class="v1-2-1">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.2.

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

```bash
$ pip install mxnet-cu92
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of v1-2-1-->

<div class="v1-1-0">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.1.

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

```bash
$ pip install mxnet-cu91==1.1.0
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of v1-1-0-->

<div class="v1-0-0">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.0.

```bash
$ pip install mxnet-cu90==1.0.0
```

</div> <!-- End of v1-0-0-->

<div class="v0-12-1">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.0.

```bash
$ pip install mxnet-cu90==0.12.1
```

Install *MXNet* 0.12.0 with GPU support using CUDA 9.0.

```bash
$ pip install mxnet-cu90==0.12.0
```

</div> <!-- End of v0-12-1-->

<div class="v0-11-0">

**Step 2**  Install *MXNet* with GPU support using CUDA 8.0.

```bash
$ pip install mxnet-cu80==0.11.0
```

</div> <!-- End of v0-11-0-->

<div class="master">

**Step 2**  Install *MXNet* with GPU support using CUDA 9.2.

**Important**: Make sure your installed CUDA version matches the CUDA version in the pip package.
Check your CUDA version with the following command:

```bash
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

```bash
$ pip install mxnet-cu92 --pre
```

Refer to [pypi for older packages](https://pypi.org/project/mxnet/).

</div> <!-- End of master-->

Refer to [#8671](https://github.com/apache/incubator-mxnet/issues/8671) for status on CUDA 9.1 support.

</div>
<div class="build-from-source">
<br/>

We provide both options to build and install MXNet yourself using [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/), and [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/).

**Option 1**

To build and install MXNet yourself using [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/), you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and install [CMake](https://cmake.org/files/v3.11/cmake-3.11.0-rc4-win64-x64.msi) if it is not already installed.
3. Download and install [OpenCV](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.1/opencv-3.4.1-vc14_vc15.exe/download).
4. Unzip the OpenCV package.
5. Set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (e.g., ```OpenCV_DIR = C:\utils\opencv\build```).
6. If you don’t have the Intel Math Kernel Library (MKL) installed, download and install [OpenBlas](https://sourceforge.net/projects/openblas/files/v0.2.20/OpenBLAS%200.2.20%20version.zip/download).
7. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories (e.g., ```OpenBLAS_HOME = C:\utils\OpenBLAS```).
8. Download and install CUDA: Install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal), and Download the base installer (e.g., ```cuda_9.1.85_win10.exe```).
9. Download and install cuDNN. To get access to the download link, register as an NVIDIA community user. Then Follow the [link](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows) to install the cuDNN.
10. Download and install [git](https://git-for-windows.github.io/).

After you have installed all of the required dependencies, build the MXNet source code:

1. Start ```cmd``` in windows.

2. Download the MXNet source code from GitHub by using following command:

```r
cd C:\
git clone https://github.com/apache/incubator-mxnet.git --recursive
```

3. Follow [this link](https://docs.microsoft.com/en-us/visualstudio/install/modify-visual-studio) to modify ```Individual components```, and check ```VC++ 2017 version 15.4 v14.11 toolset```, and click ```Modify```.

4. Change the version of the Visual studio 2017 to v14.11 using the following command (by default the VS2017 is installed in the following path):

```r
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.11
```

5. Create a build dir using the following command and go to the directory, for example:

```r
mkdir C:\build
cd C:\build
```

6. CMake the MXNet source code by using following command:

```r
cmake -G "Visual Studio 15 2017 Win64" -T cuda=9.1,host=x64 -DUSE_CUDA=1 -DUSE_CUDNN=1 -DUSE_NVRTC=1 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_LIST=Common -DCUDA_TOOLSET=9.1 -DCUDNN_INCLUDE=C:\cuda\include -DCUDNN_LIBRARY=C:\cuda\lib\x64\cudnn.lib "C:\incubator-mxnet"
```

NOTE: make sure the DCUDNN_INCLUDE and DCUDNN_LIBRARY pointing to the “include” and “cudnn.lib” of your CUDA installed location, and the ```C:\incubator-mxnet``` is the location of the source code you just git in the previous step

7. After the CMake successfully completed, compile the the MXNet source code by using following command:

```r
msbuild mxnet.sln /p:Configuration=Release;Platform=x64 /maxcpucount
```

**Option 2**

To build and install MXNet yourself using [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/), you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is not already installed, download and install it. You can download and install the free community edition. At least Update 3 of Microsoft Visual Studio 2015 is required to build MXNet from source. Upgrade via it's ```Tools -> Extensions and Updates... | Product Updates``` menu.
2. Download and install [CMake](https://cmake.org/) if it is not already installed.
3. Download and install [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download).
4. Unzip the OpenCV package.
5. Set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (```C:\opencv\build\x64\vc14``` for example). Also, you need to add the OpenCV bin directory (```C:\opencv\build\x64\vc14\bin``` for example) to the ``PATH`` variable.
6. If you don't have the Intel Math Kernel Library (MKL) installed, download and install [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/).
7. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories. Typically, you can find the directory in ```C:\Program files (x86)\OpenBLAS\```.
8. Download and install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) and [cuDNN](https://developer.nvidia.com/cudnn). To get access to the download link, register as an NVIDIA community user.
9. Set the environment variable ```CUDACXX``` to point to the ```CUDA Compiler```(```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin\nvcc.exe``` for example).
10. Set the environment variable ```CUDNN_ROOT``` to point to the ```cuDNN``` directory that contains the ```include```,  ```lib``` and ```bin``` directories (```C:\Downloads\cudnn-9.1-windows7-x64-v7\cuda``` for example).

After you have installed all of the required dependencies, build the MXNet source code:

1. Download the MXNet source code from [GitHub](https://github.com/apache/incubator-mxnet) (make sure you also download third parties submodules e.g. ```git clone --recurse-submodules```).
2. Use [CMake](https://cmake.org/) to create a Visual Studio solution in ```./build```.
3. In Visual Studio, open the solution file,```.sln```, and compile it.
These commands produce a library called ```mxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.

&nbsp;
Next, we install the ```graphviz``` library that we use for visualizing network graphs that you build on MXNet. We will also install [Jupyter Notebook](http://jupyter.readthedocs.io/) which is used for running MXNet tutorials and examples.
- Install the ```graphviz``` by downloading the installer from the [Graphviz Download Page](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).
**Note** Make sure to add the `graphviz` executable path to the PATH environment variable. Refer [here for more details](http://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)


&nbsp;
</div> <!-- End of pip -->
</div> <!-- End of GPU -->
</div> <!-- End of Python -->


<!-- START - Windows R CPU Installation Instructions -->

<div class="r">
<div class="cpu">

Install the latest version (3.5.1+) of R from [CRAN](https://cran.r-project.org/bin/windows/).
You can [build MXNet-R from source](windows_setup.html#install-mxnet-package-for-r), or you can use a pre-built binary:

```r
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
```

</div> <!-- END - Windows R CPU -->

<div class="gpu">

You can [build MXNet-R from source](windows_setup.html#install-mxnet-package-for-r), or you can use a pre-built binary:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
  options(repos = cran)
  install.packages("mxnet")
```
Change cu92 to cu80, cu90 or cu91 based on your CUDA toolkit version. Currently, MXNet supports these versions of CUDA.

</div> <!-- END of GPU -->
</div> <!-- END - Windows R -->

<div class="scala">
<div class="cpu gpu">

MXNet-Scala for Windows is not yet available.
<br>
</div> <!-- End of cpu gpu -->
</div> <!-- End of scala -->

<div class="julia perl">
<div class="cpu gpu">

Follow the installation instructions [in this guide](./windows_setup.md) to set up MXNet.

</div> <!-- End of cpu gpu -->
</div> <!-- End of julia perl -->

<div class="cpp">
<div class="cpu gpu">
<p> To build the C++ package, please refer to <a href="build_from_source.html#build-the-c-package">this guide</a>. </p>
<br/>
</div> <!-- End of cpu gpu -->
</div> <!-- End of C++ -->
</div> <!-- End of Windows -->


<!-- START - Cloud Python Installation Instructions -->

<div class="cloud">

AWS Marketplace distributes Deep Learning AMIs (Amazon Machine Image) with MXNet pre-installed. You can launch one of these Deep Learning AMIs by following instructions in the [AWS Deep Learning AMI Developer Guide](http://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html).

You can also run distributed deep learning with *MXNet* on AWS using [Cloudformation Template](https://github.com/awslabs/deeplearning-cfn/blob/master/README.md).

</div> <!-- END - Cloud Python Installation Instructions -->


<!-- DEVICES -->
<div class="devices">
  <div class="raspberry-pi">

MXNet supports the Debian based Raspbian ARM based operating system so you can run MXNet on Raspberry Pi Devices.

These instructions will walk through how to build MXNet for the Raspberry Pi and install the Python bindings for the library.

You can do a dockerized cross compilation build on your local machine or a native build on-device.

The complete MXNet library and its requirements can take almost 200MB of RAM, and loading large models with the library can take over 1GB of RAM. Because of this, we recommend running MXNet on the Raspberry Pi 3 or an equivalent device that has more than 1 GB of RAM and a Secure Digital (SD) card that has at least 4 GB of free memory.

**Cross compilation build (Experimental)**

## Docker installation
**Step 1**  Install Docker on your machine by following the [docker installation instructions](https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository).

*Note* - You can install Community Edition (CE)

**Step 2** [Optional] Post installation steps to manage Docker as a non-root user.

Follow the four steps in this [docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user) to allow managing docker containers without *sudo*.

## Build

The following command will build a container with dependencies and tools and then compile MXNet for
ARMv7. The resulting artifact will be located in `build/mxnet-x.x.x-py2.py3-none-any.whl`, copy this
file to your Raspberry Pi.

```bash
ci/build.py -p armv7
```

## Install

Create a virtualenv and install the package we created previously.

```bash
virtualenv -p `which python3` mxnet_py3
source mxnet_py3/bin/activate
pip install mxnet-x.x.x-py2.py3-none-any.whl
```


**Native Build**

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Step 1** Build the Shared Library

On Raspbian versions Wheezy and later, you need the following dependencies:

- Git (to pull code from GitHub)

- libblas (for linear algebraic operations)

- libopencv (for computer vision operations. This is optional if you want to save RAM and Disk Space)

- A C++ compiler that supports C++ 11. The C++ compiler compiles and builds MXNet source code. Supported compilers include the following:

- [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/). Make sure to use gcc 4 and not 5 or 6 as there
  are known bugs with these compilers.

Install these dependencies using the following commands in any directory:

```bash
    sudo apt-get update
    sudo apt-get -y install git cmake ninja-build build-essential g++-4.9 c++-4.9 liblapack* libblas* libopencv* libopenblas* python3-dev virtualenv
```

Clone the MXNet source code repository using the following `git` command in your home directory:
```bash
    git clone https://github.com/apache/incubator-mxnet.git --recursive
    cd incubator-mxnet
```

Build:
```bash
    mkdir -p build && cd build
    cmake \
        -DUSE_SSE=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=ON \
        -DUSE_OPENMP=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -GNinja ..
    ninja -j1
```
Some compilation units require memory close to 1GB, so it's recommended that you enable swap as
explained below and be cautious about increasing the number of jobs when building (-j)

Executing these commands start the build process, which can take up to a couple hours, and creates a file called `libmxnet.so` in the build directory.

If you are getting build errors in which the compiler is being killed, it is likely that the
compiler is running out of memory (especially if you are on Raspberry Pi 1, 2 or Zero, which have
less than 1GB of RAM), this can often be rectified by increasing the swapfile size on the Pi by
editing the file /etc/dphys-swapfile and changing the line CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024,
then running:
```bash
  sudo /etc/init.d/dphys-swapfile stop
  sudo /etc/init.d/dphys-swapfile start
  free -m # to verify the swapfile size has been increased
```

**Step 2** Install MXNet Python Bindings

To install Python bindings run the following commands in the MXNet directory:

```bash
    cd python
    pip install --upgrade pip
    pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

Alternatively you can create a whl package installable with pip with the following command:
```bash
ci/docker/runtime_functions.sh build_wheel python/ $(realpath build)
```


You are now ready to run MXNet on your Raspberry Pi device. You can get started by following the tutorial on [Real-time Object Detection with MXNet On The Raspberry Pi](http://mxnet.io/tutorials/embedded/wine_detector.html).

*Note - Because the complete MXNet library takes up a significant amount of the Raspberry Pi's limited RAM, when loading training data or large models into memory, you might have to turn off the GUI and terminate running processes to free RAM.*

</div> <!-- End of raspberry pi -->


<div class="nvidia-jetson">

# Nvidia Jetson TX family

MXNet supports the Ubuntu Arch64 based operating system so you can run MXNet on NVIDIA Jetson Devices.

These instructions will walk through how to build MXNet for the Pascal based [NVIDIA Jetson TX2](http://www.nvidia.com/object/embedded-systems-dev-kits-modules.html) and install the corresponding python language bindings.

For the purposes of this install guide we will assume that CUDA is already installed on your Jetson device.

**Install MXNet**

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Step 1** Build the Shared Library

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

Clone the MXNet source code repository using the following `git` command in your home directory:
```bash
    git clone https://github.com/apache/incubator-mxnet.git --recursive
    cd incubator-mxnet
```

Edit the Makefile to install the MXNet with CUDA bindings to leverage the GPU on the Jetson:
```bash
    cp make/crosscompile.jetson.mk config.mk
```

Edit the Mshadow Makefile to ensure MXNet builds with Pascal's hardware level low precision acceleration by editing 3rdparty/mshadow/make/mshadow.mk and adding the following after line 122:
```bash
MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1
```

Now you can build the complete MXNet library with the following command:
```bash
    make -j $(nproc)
```

Executing this command creates a file called `libmxnet.so` in the mxnet/lib directory.

**Step 2** Install MXNet Python Bindings

To install Python bindings run the following commands in the MXNet directory:

```bash
    cd python
    pip install --upgrade pip
    pip install -e .
```

Note that the `-e` flag is optional. It is equivalent to `--editable` and means that if you edit the source files, these changes will be reflected in the package installed.

Add the mxnet folder to the path:

```bash
    cd ..
    export MXNET_HOME=$(pwd)
    echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.bashrc
    source ~/.bashrc
```

You are now ready to run MXNet on your NVIDIA Jetson TX2 device.

</div> <!-- End of jetson -->
</div> <!-- End of devices -->


<!-- This # tag restarts the page and allows reuse
 of the div classes for validation sections, etc -->


# Validate MXNet Installation

<div class="linux macos">
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

Run a short *MXNet* python program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3))
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```
</div><!-- linux macos -->
</div><!-- python -->
</div><!-- cpu -->

<!-- Validate Windows CPU pip install -->

<div class="windows">
<div class="python">
<div class="cpu">
<div class="pip">

Run a short *MXNet* python program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

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
</div>

<!-- Mac OS GPU installation validation -->

<div class="macos">
<div class="python">
<div class="gpu">

<div class="pip virtualenv docker">
</br>

Will be available soon.

</div>

<div class="build-from-source">
</br>

From the MXNet root directory run: `python example/image-classification/train_mnist.py --network lenet --gpus 0` to test GPU training.

</div>

</div>
</div>
</div>

<!-- Windows GPU installation validation -->

<div class="windows">
<div class="python">
<div class="gpu">

<div class="virtualenv docker">
</br>

Will be available soon.

</div>

<div class="pip build-from-source">
</br>

From the MXNet root directory run: `python example/image-classification/train_mnist.py --network lenet --gpus 0` to test GPU training.

</div>

</div><!-- windows -->
</div><!-- python -->
</div><!-- gpu -->

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

Run a short *MXNet* python program to create a 2X3 matrix of ones *a* on a *GPU*, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3. We use *mx.gpu()*, to set *MXNet* context to be GPUs.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```
</div><!-- linux -->
</div><!-- python -->
</div><!-- gpu -->






<!-- Linux Clean up -->
<div class="linux">
<div class="python">
<div class="cpu">

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

Exit the Python terminal and mxnet/python docker container.
```python
>>> exit()
root@4919c4f58cac:/# exit
```

</div>

</div>
</div>
</div>

<!-- MacOS Clean up -->
<div class="macos">
<div class="python">
<div class="cpu">

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

<!-- Validation for cloud installation -->

<div class="cloud">

Login to the cloud instance you launched, with pre-installed *MXNet*, following the guide by corresponding cloud provider.


Start the python terminal.

```bash
$ python
```
<!-- Example Python code for CPU -->

<div class="cpu">

Run a short *MXNet* python program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3))
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
         [ 3.,  3.,  3.]], dtype=float32)
  ```

Exit the Python terminal.

```python
>>> exit()
$
```

</div>

<!-- Example Python code for CPU -->

<div class="gpu">

Run a short *MXNet* python program to create a 2X3 matrix of ones *a* on a *GPU*, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3. We use *mx.gpu()*, to set *MXNet* context to be GPUs.

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

<!-- Example R code for CPU -->

<div class="linux macos windows">
<div class="r">
<div class="cpu">

Run a short *MXNet* R program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

```r
library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.cpu())
b <- a * 2 + 1
b
```

You should see the following output:

```r
[,1] [,2] [,3]
[1,]    3    3    3
[2,]    3    3    3
```

</div>
</div>
</div>

<!-- Example R code for GPU -->

<div class="linux macos windows">
<div class="r">
<div class="gpu">

Run a short *MXNet* R program to create a 2X3 matrix of ones *a* on a *GPU*, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3. We use *mx.gpu()*, to set *MXNet* context to be GPUs.

```r
library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.gpu())
b <- a * 2 + 1
b
```

You should see the following output:

```r
[,1] [,2] [,3]
[1,]    3    3    3
[2,]    3    3    3
```

</div>
</div>
</div>



<div class="linux">
<div class="scala">

<div class="cpu gpu">
      Run the <a href="https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo">MXNet-Scala demo project</a> to validate your Maven package installation.
</div>

</div><!-- scala -->

<div class="julia perl cpp">
<div class="cpu gpu">

Will be available soon.

</div><!-- cpu gpu -->
</div><!-- julia perl cpp -->
</div><!-- linux -->

<div class="macos">
<div class="scala">
<div class="cpu gpu">
      Run the <a href="https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo">MXNet-Scala demo project</a> to validate your Maven package installation.
</div><!-- cpu gpu-->
</div><!-- scala -->
<div class="julia perl cpp">
<div class="cpu gpu">

Will be available soon.

</div><!-- cpu gpu -->
</div><!-- julia perl cpp -->
</div><!-- macos -->

<!-- Windows MXNet Installation validation -->
<div class="windows">
<div class="python">
<div class="cpu">

<div class="build-from-source virtualenv docker">
<br/>
Will be available soon.
</div>

</div>
</div>

<div class="scala julia perl cpp">
<div class="cpu gpu">

Will be available soon.

</div>
</div>
</div>
<!-- End Windows Installation validation -->

<br/>
<!-- Download -->

# Source Download

<a href="download.html">Download</a> your required version of MXNet.
