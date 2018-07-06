# Installing MXNet on Ubuntu

The following installation instructions are for installing MXNet on computers running **Ubuntu 16.04**. Support for later versions of Ubuntu is [not yet available](#contributions).
<hr>

## Contents

* [CUDA Dependencies](#cuda-dependencies)
* [Quick Installation](#quick-installation)
    * [Python](#install-mxnet-for-python)
    * [pip Packages](#pip-package-availability)
* [Standard Installation](#standard-installation)
* [Installing Language Packages](#installing-language-packages-for-mxnet)
    * [R](#install-the-mxnet-package-for-r)
    * [Julia](#install-the-mxnet-package-for-julia)
    * [Scala](#install-the-mxnet-package-for-scala)
    * [Perl](#install-the-mxnet-package-for-perl)
  * [Contributions](#contributions)
  * [Next Steps](#next-steps)

<hr>

## CUDA Dependencies

If you plan to build with GPU, you need to set up the environment for CUDA and cuDNN.

First, download and install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit). CUDA 9.2 is recommended.

Then download [cuDNN 7.1.4](https://developer.nvidia.com/cudnn).

Unzip the file and change to the cuDNN root directory. Move the header and libraries to your local CUDA Toolkit folder:

```bash
    tar xvzf cudnn-9.2-linux-x64-v7.1
    sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    sudo ldconfig
```

<hr>


## Quick Installation

### Install MXNet for Python

#### Dependencies

The following scripts will install Ubuntu 16.04 dependencies for MXNet Python development.

```bash
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_core.sh
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_python.sh
sudo ./ubuntu_core.sh
sudo ./ubuntu_python.sh
```

Using the latest MXNet with CUDA 9.2 package is recommended for the fastest training speeds with MXNet.

**Recommended for training:**
```bash
pip install mxnet-cu92
```

**Recommended for inference:**
```bash
pip install mxnet-cu92mkl
```

Alternatively, you can use the table below to select the package that suits your purpose.

| MXNet Version | Basic | CUDA | MKL | CUDA/MKL |
|-|-|-|-|-|
| Latest | mxnet | mxnet-cu92 | mxnet-mkl | mxnet-cu92mkl |


#### pip Package Availability

The following table presents the pip packages that are recommended for each version of MXNet.

<!-- Must find sol'n for both github and website; image in the meantime
| Package / MXNet Version | 1.2.1 | 1.1.0 | 1.0.0 | 0.12.1 | 0.11.0 |
|-|-|-|-|-|-|
| mxnet-cu92mkl | :white_check_mark:<i class="fas fa-check"></i> | :x: | :x: | :x: | :x: |
| mxnet-cu92 | :white_check_mark:<i class="fas fa-check"></i> | :x: | :x: | :x: | :x: |
| mxnet-cu90mkl | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| mxnet-cu90 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| mxnet-cu80mkl | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| mxnet-cu80 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| mxnet-mkl | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| mxnet | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
-->

![pip package table](https://user-images.githubusercontent.com/5974205/42119928-362ad5ba-7bc7-11e8-97de-dba8fd099c90.png)

To install an older version of MXNet with one of the packages in the previous table add `==` with the version you require. For example for version 1.1.0 of MXNet with CUDA 8, you would use `pip install mxnet-cu80==1.1.0`.

<hr>


## Standard installation

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Note:** To change the compilation options for your build, edit the ```make/config.mk``` file prior to building MXNet. More information on this is mentioned in the different language package instructions.

### Build the Shared Library

On Ubuntu versions 13.10 or later, you need the following dependencies:

**Step 1:** Install build tools and git.
```bash
    sudo apt-get update
    sudo apt-get install -y build-essential git
```

**Step 2:** Install OpenBLAS.

*MXNet* uses [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library for accelerated numerical computations on CPU machine. There are several flavors of BLAS libraries - [OpenBLAS](http://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/) and [MKL](https://software.intel.com/en-us/intel-mkl). In this step we install OpenBLAS. You can choose to install ATLAS or MKL.

```bash
    sudo apt-get install -y libopenblas-dev
```

**Step 3:** Install OpenCV.

*MXNet* uses [OpenCV](http://opencv.org/) for efficient image loading and augmentation operations.

```bash
    sudo apt-get install -y libopencv-dev
```

**Step 4:** Download MXNet sources and build MXNet core shared library.

If building on CPU:

```bash
    git clone --recursive https://github.com/apache/incubator-mxnet.git
    cd mxnet
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas
```

If building on GPU:

```bash
    git clone --recursive https://github.com/apache/incubator-mxnet.git
    cd mxnet
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
```

*Note* - USE_OPENCV and USE_BLAS are make file flags to set compilation options to use OpenCV and BLAS library. You can explore and use more compilation options in `make/config.mk`.

Executing these commands creates a library called ```libmxnet.so```.

Next, you may optionally install ```graphviz``` library that is used for visualizing network graphs you build on MXNet. You may also install [Jupyter Notebook](http://jupyter.readthedocs.io/) which is used for running MXNet tutorials and examples.

```bash
    sudo apt-get install -y python-pip
    sudo pip install graphviz
    sudo pip install jupyter
```
<hr>


## Installing Language Packages for MXNet

After you have installed the MXNet core library. You may install MXNet interface packages for the programming language of your choice:
- [Scala](#install-the-mxnet-package-for-scala)
- [R](#install-the-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- [Perl](#install-the-mxnet-package-for-perl)


### Install the MXNet Package for Scala

To use the MXNet-Scala package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Scala Setup Instructions](./scala_setup.md).

If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial](../tutorials/scala/mxnet_scala_on_intellij.md) instead.
<hr>

### Install the MXNet Package for R

#### MXNet-R Dependencies

For users of R on Ubuntu operating systems, MXNet provides a set of Git Bash scripts that installs all of the required MXNet dependencies and the MXNet library. The scripts install MXNet in your home folder ```~/mxnet```.

MXNet requires R-version to be 3.2.0 and above. If you are running an earlier version of R, run below commands to update your R version, before running the installation script.

```bash
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    sudo add-apt-repository ppa:marutter/rdev

    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install r-base r-base-dev
```

To install MXNet for R:

```bash
    # Clone mxnet repository. In terminal, run the commands WITHOUT "sudo"
    git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive

    cd ~/mxnet
    cp make/config.mk .
    # If building with GPU, add configurations to config.mk file:
    echo "USE_CUDA=1" >>config.mk
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
    echo "USE_CUDNN=1" >>config.mk

    cd ~/mxnet/setup-utils
    bash install-mxnet-ubuntu-r.sh
```
The installation script to install MXNet for R can be found [here](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-ubuntu-r.sh).

Run the following commands to install the MXNet dependencies and build the MXNet R package.

```r
    Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
```
```bash
    cd R-package
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    cd ..
    make rpkg
```

**Note:** R-package is a folder in the MXNet source.

These commands create the MXNet R package as a tar.gz file that you can install as an R package. To install the R package, run the following command, use your MXNet version number:

```bash
    R CMD INSTALL mxnet_current_r.tar.gz
```
<hr>


### Install the MXNet Package for Julia

The MXNet package for Julia is hosted in a separate repository, MXNet.jl, which is available on [GitHub](https://github.com/dmlc/MXNet.jl). To use Julia binding it with an existing libmxnet installation, set the ```MXNET_HOME``` environment variable by running the following command:

```bash
    export MXNET_HOME=/<path to>/libmxnet
```

The path to the existing libmxnet installation should be the root directory of libmxnet. In other words, you should be able to find the ```libmxnet.so``` file at ```$MXNET_HOME/lib```. For example, if the root directory of libmxnet is ```~```, you would run the following command:

```bash
    export MXNET_HOME=/~/libmxnet
```

You might want to add this command to your ```~/.bashrc``` file. If you do, you can install the Julia package in the Julia console using the following command:

```julia
    Pkg.add("MXNet")
```

For more details about installing and using MXNet with Julia, see the [MXNet Julia documentation](http://dmlc.ml/MXNet.jl/latest/user-guide/install/).
<hr>


## Install the MXNet Package for Scala

To use the MXNet-Scala package, you can acquire the Maven package as a dependency.

Further information is in the [MXNet-Scala Setup Instructions](./scala_setup.md).

If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Scala on IntelliJ tutorial](../tutorials/scala/mxnet_scala_on_intellij.md) instead.


### Install the MXNet Package for Perl

Before you build MXNet for Perl from source code, you must complete [building the shared library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Perl package:

```bash
    sudo apt-get install libmouse-perl pdl cpanminus swig libgraphviz-perl
    cpanm -q -L "${HOME}/perl5" Function::Parameters Hash::Ordered PDL::CCS

    MXNET_HOME=${PWD}
    export LD_LIBRARY_PATH=${MXNET_HOME}/lib
    export PERL5LIB=${HOME}/perl5/lib/perl5

    cd ${MXNET_HOME}/perl-package/AI-MXNetCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install

    cd ${MXNET_HOME}/perl-package/AI-NNVMCAPI/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install

    cd ${MXNET_HOME}/perl-package/AI-MXNet/
    perl Makefile.PL INSTALL_BASE=${HOME}/perl5
    make install
```
<hr>

## Contributions

You are more than welcome to contribute easy installation scripts for other operating systems and programming languages. See the [community contributions page](../community/contribute.html) for further information.

## Next Steps

* [Tutorials](../tutorials/index.html)
* [How To](../faq/index.html)
* [Architecture](../architecture/index.html)


<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.1.0/css/all.css" integrity="sha384-lKuwvrZot6UHsBSfcMvOkWwlCMgc0TaWr+30HWe3a4ltaBwTZhyTEggF5tJv8tbt" crossorigin="anonymous">
