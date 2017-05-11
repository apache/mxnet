# Installing MXNet on Amazon Linux

**NOTE:** For MXNet with Python installation, please refer to the [new install guide](http://mxnet.io/get_started/install.html).

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Note:** To change the compilation options for your build, edit the ```make/config.mk``` file and submit a build request with the ```make``` command.

## Build the Shared Library
On Amazon Linux, you need the following dependencies:

- Git (to pull code from GitHub)

- libatlas-base-dev (for linear algebraic operations)

- libopencv-dev (for computer vision operations)

Install these dependencies using the following commands:

```bash
      # CMake is required for installing dependencies.
      sudo yum install -y cmake

      # Set appropriate library path env variables
      echo 'export PATH=/usr/local/bin:$PATH' >> ~/.profile
      echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.profile
      echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.profile
      echo '. ~/.profile' >> ~/.bashrc
      source ~/.profile

      # Install gcc-4.8/make and other development tools on Amazon Linux
      # Reference: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compile-software.html
      # Install Python, Numpy, Scipy and set up tools.
      sudo yum groupinstall -y "Development Tools"
      sudo yum install -y python27 python27-setuptools python27-tools python-pip
      sudo yum install -y python27-numpy python27-scipy python27-nose python27-matplotlib graphviz

      # Install OpenBLAS at /usr/local/openblas
      git clone https://github.com/xianyi/OpenBLAS
      cd OpenBLAS
      make FC=gfortran -j $(($(nproc) + 1))
      sudo make PREFIX=/usr/local install
      cd ..

      # Install OpenCV at /usr/local/opencv
      git clone https://github.com/opencv/opencv
      cd opencv
      mkdir -p build
      cd build
      cmake -D BUILD_opencv_gpu=OFF -D WITH_EIGEN=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
      sudo make PREFIX=/usr/local install

      # Install Graphviz for visualization and Jupyter notebook for running examples and tutorials
      sudo pip install graphviz
      sudo pip install jupyter

      # Export env variables for pkg config
      export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```
After installing the dependencies, use the following command to pull the MXNet source code from GitHub

```bash
    # Get MXNet source code
    git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive
    # Move to source code parent directory
    cd ~/mxnet
    cp make/config.mk .
    echo "USE_BLAS=openblas" >>config.mk
    echo "ADD_CFLAGS += -I/usr/include/openblas" >>config.mk
    echo "ADD_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs" >>config.mk
```

If building with ```GPU``` support, run below commands to add GPU dependency configurations to config.mk file:

```bash
    echo "USE_CUDA=1" >>config.mk
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
    echo "USE_CUDNN=1" >>config.mk
```

Then build mxnet:

```bash
    make -j$(nproc)
```

Executing these commands creates a library called ```libmxnet.so```


&nbsp;

We have installed MXNet core library. Next, we will install MXNet interface package for the programming language of your choice:
- [R](#install-the-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- [Scala](#install-the-mxnet-package-for-scala)
- [Perl](#install-the-mxnet-package-for-perl)

## Install the MXNet Package for R
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

## Install the MXNet Package for Julia
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

## Install the MXNet Package for Scala

There are two ways to install the MXNet package for Scala:

* Use the prebuilt binary package

* Build the library from source code

### Use the Prebuilt Binary Package
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

### Build the Library from Source Code
Before you build MXNet for Scala from source code, you must complete [building the shared library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Scala package:

```bash
  make scalapkg
```

This command creates the JAR files for the assembly, core, and example modules. It also creates the native library in the ```native/{your-architecture}/target directory```, which you can use to cooperate with the core module.

To install the MXNet Scala package into your local Maven repository, run the following command from the MXNet source root directory:

```bash
  make scalainstall
```

## Install the MXNet Package for Perl

Before you build MXNet for Scala from source code, you must complete [building the shared library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Scala package:

```bash
    ## install PDL, Graphviz, Mouse, App::cpanminus, swig via yum before running these commands
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

**Note -** You are more than welcome to contribute easy installation scripts for other operating systems and programming languages, see [community page](http://mxnet.io/community/index.html) for contributors guidelines.

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
