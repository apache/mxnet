# Installing MXNet on Ubuntu

**NOTE:** For MXNet with Python installation, please refer to the [new install guide](http://mxnet.io/get_started/install.html).

MXNet currently supports Python, R, Julia, Scala, and Perl. For users of R on Ubuntu operating systems, MXNet provides a set of Git Bash scripts that installs all of the required MXNet dependencies and the MXNet library.

The simple installation scripts set up MXNet for Python and R on computers running Ubuntu 12 or later. The scripts install MXNet in your home folder ```~/mxnet```.

## Prepare environment for GPU Installation

If you plan to build with GPU, you need to set up the environment for CUDA and CUDNN.

First, download and install [CUDA 8 toolkit](https://developer.nvidia.com/cuda-toolkit).

Then download [cudnn 6](https://developer.nvidia.com/cudnn).

Unzip the file and change to the cudnn root directory. Move the header and libraries to your local CUDA Toolkit folder:

```bash
    tar xvzf cudnn-8.0-linux-x64-v6.0.tgz
    sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    sudo ldconfig
```

Finally, add configurations to config.mk file:

```bash
    cp make/config.mk .
```

## Quick Installation

### Install MXNet for R

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

## Standard installation

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Note:** To change the compilation options for your build, edit the ```make/config.mk``` file and submit a build request with the ```make``` command.

### Build the Shared Library

On Ubuntu versions 13.10 or later, you need the following dependencies:

**Step 1** Install build tools and git.
```bash
    sudo apt-get update
    sudo apt-get install -y build-essential git
```

**Step 2** Install OpenBLAS.

*MXNet* uses [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library for accelerated numerical computations on CPU machine. There are several flavors of BLAS libraries - [OpenBLAS](http://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/) and [MKL](https://software.intel.com/en-us/intel-mkl). In this step we install OpenBLAS. You can choose to install ATLAS or MKL.

```bash
    sudo apt-get install -y libopenblas-dev
```

**Step 3** Install OpenCV.

*MXNet* uses [OpenCV](http://opencv.org/) for efficient image loading and augmentation operations.

```bash
    sudo apt-get install -y libopencv-dev
```

**Step 4** Download MXNet sources and build MXNet core shared library.

If building on CPU:

```bash
    git clone --recursive https://github.com/dmlc/mxnet
    cd mxnet
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas
```

If building on GPU:

```bash
    git clone --recursive https://github.com/dmlc/mxnet
    cd mxnet
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
```

*Note* - USE_OPENCV and USE_BLAS are make file flags to set compilation options to use OpenCV and BLAS library. You can explore and use more compilation options in `make/config.mk`.

Executing these commands creates a library called ```libmxnet.so```.

Next, we install ```graphviz``` library that we use for visualizing network graphs you build on MXNet. We will also install [Jupyter Notebook](http://jupyter.readthedocs.io/) used for running MXNet tutorials and examples.

```bash
    sudo apt-get install -y python-pip
    sudo pip install graphviz
    sudo pip install Jupyter
```

&nbsp;

We have installed MXNet core library. Next, we will install MXNet interface package for programming language of your choice:
- [R](#install-the-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- [Scala](#install-the-mxnet-package-for-scala)
- [Perl](#install-the-mxnet-package-for-perl)

### Install the MXNet Package for R

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

### Install the MXNet Package for Scala
There are two ways to install the MXNet package for Scala:

* Use the prebuilt binary package

* Build the library from source code

#### Use the Prebuilt Binary Package
For Linux users, MXNet provides prebuilt binary packages that support computers with either GPU or CPU processors. To download and build these packages using ```Maven```, change the ```artifactId``` in the following Maven dependency to match your architecture:

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_<system architecture></artifactId>
  <version>0.1.1</version>
</dependency>
```

For example, to download and build the 64-bit CPU-only version for Linux, use:

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_2.10-linux-x86_64-cpu</artifactId>
  <version>0.1.1</version>
</dependency>
```

If your native environment differs slightly from the assembly package, for example, if you use the openblas package instead of the atlas package, it's better to use the mxnet-core package and put the compiled Java native library in your load path:

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-core_2.10</artifactId>
  <version>0.1.1</version>
</dependency>
```

#### Build the Library from Source Code
Before you build MXNet for Scala from source code, you must complete [building the shared library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Scala package:

```bash
    make scalapkg
```

This command creates the JAR files for the assembly, core, and example modules. It also creates the native library in the ```native/{your-architecture}/target directory```, which you can use to cooperate with the core module.

To install the MXNet Scala package into your local Maven repository, run the following command from the MXNet source root directory:

```bash
    make scalainstall
```
### Install the MXNet Package for Perl

Before you build MXNet for Perl from source code, you must complete [building the shared library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Perl package:

```bash
    sudo apt-get install libmouse-perl pdl cpanminus swig libgraphviz-perl
    cpanm -q -L "${HOME}/perl5" Function::Parameters

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

**Note - ** You are more than welcome to contribute easy installation scripts for other operating systems and programming languages, see [community page](http://mxnet.io/community/index.html) for contributors guidelines.

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
