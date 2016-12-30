# Installing MXNet on OS X (Mac)

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code.
2. Install the supported language-specific packages for MXNet.

**Note:** To change the compilation options for your build, edit the ```make/config.mk``` file and submit a build request with the ```make``` command.

## Using the installation script

Install [Homebrew](http://brew.sh/)  to get the dependencies required for MXNet, with the following commands:

```bash
	# Paste this command in Mac terminal to install Homebrew
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

	# Insert the Homebrew directory at the top of your PATH environment variable
	export PATH=/usr/local/bin:/usr/local/sbin:$PATH
```

Initialize the git submodules and fetch them:

```
  git submodule init
  git submodule update
```

Then you can execute the `setup-utils/install-mxnet-macosx.sh` script
[1](https://github.com/dmlc/mxnet/tree/master/setup-utils/setup-utils/install-mxnet-macosx.sh).

After that, you should have the mxnet library in the `lib` directory:
```
  mxnet$ ls lib/
  libmxnet.a  libmxnet.so
```

Now you can skip to "Install the MXNet Package for Python" down below if you plan to use mxnet with Python.

--

## Manual installation steps

Alternatively you can follow through with the following manual instructions:

You will need to install:

- openblas and homebrew/science (for linear algebraic operations)
- OpenCV (for computer vision operations)

```bash
	brew update
	brew install pkg-config
	brew install graphviz
	brew install openblas
	brew tap homebrew/science
	brew install opencv
	# For getting pip
	brew install python
	# For visualization of network graphs
	pip install graphviz
	# Jupyter notebook
	pip install jupyter
```
After you have installed the dependencies, use one of the following options to pull the MXNet source code from Git and build MXNet. Both options produce a library called ```libmxnet.so```.

### Prepare Environment for GPU Installation

If you plan to build with GPU, you need to set up environemtn for CUDA and cuDNN.

First download and install [CUDA 8 toolkit](https://developer.nvidia.com/cuda-toolkit).

Once you have the CUDA Toolkit installed you will need to setup the required environment variables by adding the following to your ~/.bash_profile file:

```bash
    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH="$CUDA_HOME/lib:$DYLD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
```

Reload ~/.bash_profile file and install dependecies:
```bash
    . ~/.bash_profile
    brew install coreutils
    brew tap caskroom/cask
```

Then download [cuDNN 5](https://developer.nvidia.com/cudnn).

Unzip the file and change to cudnn root directory. Move the header files and libraries to your local CUDA Toolkit folder:

```bash
    $ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-8.0/include/
    $ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-8.0/lib
    $ sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn* /usr/local/cuda/lib/
```

Now we can start to build MXNet.

**Option 1** Use the following commands to pull MXNet source code and build MXNet. The file called ```osx.mk``` has the configuration required for building MXNet on OS X. First copy ```make/osx.mk``` into ```config.mk```, which is used by the ```make``` command:

```bash
    git clone --recursive https://github.com/dmlc/mxnet
    cd mxnet
    cp make/osx.mk ./config.mk
    echo "USE_BLAS = openblas" >> ./config.mk
    echo "ADD_CFLAGS += -I/usr/local/opt/openblas/include" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/" >> ./config.mk
    make -j$(sysctl -n hw.ncpu)
```

If building with GPU, add the following configuration to config.mk and build:
```bash
    echo "USE_CUDA = 1" >> ./config.mk
    echo "USE_CUDA_PATH = /usr/local/cuda" >> ./config.mk
    echo "USE_CUDNN = 1" >> ./config.mk
    make
```
**Note:** To change build parameters, edit ```config.mk```.

**Option 2**
To generate an [Xcode](https://en.wikipedia.org/wiki/Xcode) project from MXNet source code, use the ```cmake``` command. Then, build MXNet using the Xcode IDE.

```bash
    mkdir build; cd build

    cmake -G Xcode -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" -DUSE_OPENMP="OFF" -DUSE_CUDNN="OFF" -DUSE_CUDA="OFF" -DBLAS=MKL ..
```

After running the ```cmake``` command, use Xcode to open ```mxnet.xcodeproj```, change the following build settings flags, and build the project:

1. Link-Time Optimization = Yes
2. Optimisation Level = Fastest[-O3]

Both sets of commands produce a library called ```libmxnet.so```.


&nbsp;

We have installed MXNet core library. Next, we will install MXNet interface package for the programming language of your choice:
- [Python](#install-the-mxnet-package-for-python)
- [R](#install-the-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- [Scala](#install-the-mxnet-package-for-scala)

## Install the MXNet Package for Python

You need the following dependencies:

- Python version 2.7 or later.
- NumPy (to provide scientific computing operations).

To check if Python is already installed run below command and you should be able to see which version of Python is installed on your machine.

```bash
	# Check if Python is already installed.
	python --version
	# Install Python if not already installed.
	brew install python
	# Install Numpy
	brew install numpy
```

Next, we install Python package interface for MXNet. You can find the Python interface package for [MXNet on GitHub](https://github.com/dmlc/mxnet/tree/master/python/mxnet).

```bash
    # Assuming you are in root mxnet source code folder
    cd python
    sudo python setup.py install
```
Done! We have installed MXNet with Python interface. Run below commands to verify our installation is successful.
```bash
    # Open Python terminal
    python

    # You should be able to import mxnet library without any issues.
    >>> import mxnet as mx;
    >>> a = mx.nd.ones((2, 3));
    >>> print ((a*2).asnumpy());
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
```
We actually did a small tensor computation using MXNet! You are all set with MXNet on your Mac.

## Install the MXNet Package for R
You have 2 options:
1. Building MXNet with the Prebuilt Binary Package
2. Building MXNet from Source Code

### Building MXNet with the Prebuilt Binary Package

For OS X (Mac) users, MXNet provides a prebuilt binary package for CPUs. The prebuilt package is updated weekly. You can install the package directly in the R console using the following commands:

```r
	install.packages("drat", repos="https://cran.rstudio.com")
	drat:::addRepo("dmlc")
	install.packages("mxnet")
```

### Building MXNet from Source Code

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
	R CMD INSTALL mxnet_0.7.tar.gz
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
Before you build MXNet for Scala from source code, you must complete [building the shared library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Scala package:

```bash
    make scalapkg
```

This command creates the JAR files for the assembly, core, and example modules. It also creates the native library in the ```native/{your-architecture}/target directory```, which you can use to cooperate with the core module.

To install the MXNet Scala package into your local Maven repository, run the following command from the MXNet source root directory:

```bash
    make scalainstall
```

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
