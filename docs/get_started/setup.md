Overview
============
You can run MXNet on Amazon Linux, Ubuntu/Debian, OS X, and Windows operating systems. MXNet supports multiple programming languages. If you are running Python on Amazon Linux or Ubuntu, you can use Git Bash scripts to quickly install the MXNet libraries and all dependencies. To use the Git Bash scripts so you can get started with MXNet quickly, skip to [Quick Installation](#quick-installation).  If you are using other languages or operating systems, keep reading.

 
This topic covers the following:



- [Prerequisites for using MXNet](#prerequisites)



- [Installing MXNet](#installing-mxnet)



- [Common installation problems](#common-installation-problems)



If you encounter problems with the instructions on this page and don't find a solution in [Common installation problems](#common-installation-problems), ask questions at [mxnet/issues](https://github.com/dmlc/mxnet/issues). If you can fix the problem, send a
pull request. For details, see [contribution guidelines](http://mxnet.io/community/index.html).

# Prerequisites

This section lists the basic requirements for running MXNet, requirements for using it with GPUs, and requirements to support computer vision and image augmentation.

## Minimum Requirements

You must have the following:

-  A C++ compiler that supports C++ 11. The C++ compiler compiles and builds MXNet source code. Supported compilers include the following:

  - [G++ (4.8 or later)](https://gcc.gnu.org/gcc-4.8/)

 -   [Clang](http://clang.llvm.org/)

- A [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (Basic Linear Algebra Subprograms) library.
  BLAS libraries contain routines that provide the standard building blocks for performing basic vector and matrix operations. You need a BLAS library to perform basic linear algebraic operations. Supported BLAS libraries include the following:
  * [libblas](http://www.netlib.org/blas/)
  * [openblas](http://www.openblas.net/)
  * [Intel MKL](https://software.intel.com/en-us/node/528497)

## Requirements for Using GPUs

* A GPU with support for Compute Capability 2.0 or higher.
  Compute capability describes which features are supported by CUDA hardware. For a list of features supported by each compute capability, see [CUDA Version features and specifications](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications). For a list of features supported by NVIDIA GPUs, see [CUDA GPUs](https://developer.nvidia.com/cuda-gpus). 
* The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 7.0 or higher.
  The CUDA Toolkit is an environment that allows MXNet to run on NVIDIA GPUs. It includes a compiler, math libraries, and debugging tools. To download the latest version, see [CUDA Toolkit download page](https://developer.nvidia.com/cuda-toolkit).
* The CuDNN (CUDA Deep Neural Network) library. 
  The CuDNN library accelerates GPU computation by providing low-level GPU performance tuning. To download the latest version, see [CUDA Deep Neural Network](https://developer.nvidia.com/cudnn).

## Requirements for Computer Vision and Image Augmentation Support

If you need to support computer vision and image augmentation, you need 
[OpenCV](http://opencv.org/).
The Open Source Computer Vision (OpenCV) library contains programming functions for computer vision and image augmentation. For more information, see [OpenCV](https://en.wikipedia.org/wiki/OpenCV).

# Cloud Setup
You can start using MXNet on cloud with MXNet pre-installed. Refer below for more details.
## Preconfigured Amazon Machine Images(AMI) with AWS
Here is a link to a blog by Jeff Barr illustrating how to setup an Amazon Machine Image(AMI) that supports both MXNet and other popular deep learning frameworks.
* [P2 and Deep Learning Blog](https://aws.amazon.com/blogs/aws/new-p2-instance-type-for-amazon-ec2-up-to-16-gpus/)
* [Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB)

## Using MXNet on multiple instances with AWS
To scale up on AWS GPU instances using a CloudFormation template, you can follow the instructions linked in the blog below.
* [CloudFormation Template AWS Blog](https://aws.amazon.com/blogs/compute/distributed-deep-learning-made-easy/)

# Installing MXNet

You can run MXNet on Amazon Linux, Ubuntu/Debian, OS X, and Windows operating systems. MXNet currently supports the Python, R, Julia, and Scala languages.
If you are running Python on Amazon Linux or Ubuntu, you can use Git Bash scripts to quickly install the MXNet libraries and all dependencies. If you are using other languages or operating systems, skip to [Standard Installation](#standard-installation).

## Quick Installation

For users of Python on Amazon Linux and Ubuntu operating systems, MXNet provides a set of Git Bash scripts that installs all of the required MXNet dependencies and the MXNet library.

To contribute easy installation scripts for other operating systems and programming languages, see [community page](http://mxnet.io/community/index.html).

### Quick Installation on Ubuntu

The simple installation scripts set up MXNet for Python and R on computers running Ubuntu 12 or later. The scripts install MXNet in your home folder ```~/mxnet```.

To clone the MXNet source code repository to your computer, use ```git```. 
```bash
# Install git if not already installed.
sudo apt-get update
sudo apt-get -y install git
```

Clone the MXNet source code repository to your computer, run the installation script, and refresh the environment variables. In addition to installing MXNet, the script installs all MXNet dependencies: ```Numpy```, ```LibBLAS``` and ```OpenCV```.
It takes around 5 minutes to complete the installation.

```bash
# Clone mxnet repository. In terminal, run the commands WITHOUT "sudo"
git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive


# Install MXNet for Python with all required dependencies
cd ~/mxnet/setup-utils
bash install-mxnet-ubuntu-python.sh

# We have added MXNet Python package path in your ~/.bashrc. 
# Run the following command to refresh environment variables.
$ source ~/.bashrc
```

You can view the installation script we just used to install MXNet for Python [here](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-ubuntu-python.sh).

# Install MXNet for R with all required dependencies

To install MXNet for R:

```bash
cd ~/mxnet/setup-utils
bash install-mxnet-ubuntu-r.sh
```
The installation script to install MXNet for R can be found [here](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-ubuntu-r.sh).

If you are unable to install MXNet with the Bash script, see the following detailed installation instructions.


### Quick Installation on Amazon Linux

The simple installation scripts set up MXNet for Python on computers running Amazon Linux. The scripts install MXNet in your home folder ```~/mxnet```.

To clone the MXNet source code repository to your computer, use ```git```. 
```bash
# Install git if not already installed.
sudo yum -y install git-all
```

Clone the MXNet source code repository to your computer, run the installation script, and refresh the environment variables. In addition to installing MXNet, the script installs all MXNet dependencies: ```Numpy```, ```OpenBLAS``` and ```OpenCV```.
It takes around 5 minutes to complete the installation.

```bash
# Clone mxnet repository. In terminal, run the commands WITHOUT "sudo"
git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive

# Install MXNet for Python with all required dependencies
cd ~/mxnet/setup-utils
bash install-mxnet-amz-linux.sh

# We have added MXNet Python package path in your ~/.bashrc. 
# Run the following command to refresh environment variables.
$ source ~/.bashrc
```

You can view the installation script [here](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-amz-linux.sh).
If you are unable to install MXNet with the Bash script, see the following detailed installation instructions.

## Standard Installation

MXNet runs on Ubuntu/Debian, Amazon Linux, RHEL, OS X (Mac), and Windows operating systems. MXNet currently supports Python, R, Julia, and Scala. The following procedures describe how to install MXNet on each of these operating systems and how to install the language-specific packages.

Installing MXNet is a two-step process:

1. Build the shared library from the MXNet C++ source code. 
2. Install the supported language-specific packages for MXNet. 

### Step 1. Build the Shared Library

The first step in installing MXNet is compiling the MXNet C++ source code and building the shared MXNet library. This section provides instructions for compiling and building the MXNet shared library on the following operating systems:

- Ubuntu/Debian

- OS X (Mac)

- Windows


**Note:** To change the compilation options for your build, edit the ```make/config.mk``` file and submit a build request with the ```make``` command.

#### Building MXNet on Ubuntu/Debian
On Ubuntu versions 13.10 or later, you need the following dependencies:

- Git (to pull code from GitHub)

- libatlas-base-dev (for linear algebraic operations)

- libopencv-dev (for computer vision operations)

Install these dependencies using the following commands:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev
```

After you have downloaded and installed the dependencies, use the following commands to pull the MXNet source code from GitHub and build MXNet:

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; make -j$(nproc)
```

Executing these commands creates a library called ```libmxnet.so```.

#### Building MXNet on OS X (Mac)

On OS X, you need the following dependencies:

- [Homebrew](http://brew.sh/) (to install dependencies)

- Git (to pull code from GitHub)

- Homebrew/science (for linear algebraic operations)

- OpenCV (for computer vision operations)

Install the dependencies with the following commands:

```bash
# Paste this command in Mac terminal to install Homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
And then: 
```bash
	brew update
	brew install pkg-config
	brew install git
	brew tap homebrew/science
	brew info opencv
	brew install opencv
```
After you have installed the dependencies, use one of the following options to pull the MXNet source code from Git and build MXNet. Both options produce a library called ```libmxnet.so```.

**Option 1**
Use the following commands to pull MXNet source code and build MXNet. The file called ```osx.mk``` has the configuration required for building MXNet on OS X. First copy ```make/osx.mk``` into ```config.mk```, which is used by the ```make``` command:

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; cp make/osx.mk ./config.mk; make -j$(sysctl -n hw.ncpu)
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

#### Building MXNet on Windows

On Windows, you can download and install the prebuilt MXNet package, or download, build, and install MXNet yourself.

##### Installing the Prebuilt Package on Windows
MXNet provides a prebuilt package for Windows. The prebuilt package includes the MXNet library, all of the dependent third-party libraries, a sample C++ solution for Visual Studio, and the Python installation script. To install the prebuilt package:

1. Download the latest prebuilt package from the [Releases](https://github.com/dmlc/mxnet/releases) tab  of MXNet. 
   There are two versions. One with GPU support (using CUDA and CUDNN v3), and one without GPU support. Choose the version that suits your hardware configuration. For more information on which version works on each hardware configuration, see [Requirements for GPU](#Requirements-for-Using-GPUs).
2. Unpack the package into a folder, with an appropriate name, such as ```D:\MXNet```.
3. Open the folder, and install the package by double-clicking ```setupenv.cmd```. This sets up all of the environment variables required by MXNet.
4. Test the installation by opening the provided sample C++ Visual Studio solution and building it.


&nbsp;
This produces a library called ```libmxnet.dll```.

##### Building and Installing Packages on Windows

To build and install MXNet yourself, you need the following dependencies:



- [Microsoft Visual Studio 2013](https://www.visualstudio.com/downloads/)



- [Visual C++ Compiler Nov 2013 CTP](https://www.microsoft.com/en-us/download/details.aspx?id=41151) (to provide C++ support)


- [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download) (for computer vision operations)


- [OpenBlas](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download) (for linear algebraic operations)



- [CuDNN](https://developer.nvidia.com/cudnn) (to provide a Deep Neural Network library)

Install the required dependencies:

1. If [Microsoft Visual Studio 2013](https://www.visualstudio.com/downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Install [Visual C++ Compiler Nov 2013 CTP](https://www.microsoft.com/en-us/download/details.aspx?id=41151).
3. Back up all of the files in the ```C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC``` folder to a different location.
4. Copy all of the files in the ```C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP``` folder (or the folder where you extracted the zip archive) to the ```C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC``` folder, and overwrite all existing files. 
5. Download and install [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download).
6. Unzip the OpenCV package.
7. Set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory```.
8. If you don't have the Intel Math Kernel Library (MKL) installed, download and install [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/). 
9. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories. Typically, you can find the directory in ```C:\Program files (x86)\OpenBLAS\```.
10. Download and install [CuDNN](https://developer.nvidia.com/cudnn). To get access to the download link, register as an NVIDIA community user.

After you have installed all of the required dependencies, build the MXNet source code:

1. Download the MXNet source code from [GitHub](https://github.com/dmlc/mxnet).
2. Use [CMake](https://cmake.org/) to create a Visual Studio solution in ```./build```.
3. In Visual Studio, open the solution file,```.sln```, and compile it.
These commands produce a library called ```mxnet.dll``` in the ```./build/Release/``` or ```./build/Debug folder```.

### Install the Language-specific Packages
Now install the packages for the programming language you want to use with MXNet. This section provides instructions for downloading and installing the following languages:

* Python

* R

* Julia

* Scala

#### Installing the MXNet Package for Python
You can find the Python interface package for [MXNet on GitHub](https://github.com/dmlc/mxnet/tree/master/python/mxnet).

You need the following dependencies:

* [Python](https://www.python.org/downloads/) version 2.7 or later.

* NumPy (to provide scientific computing operations). 
If you need to install Python, download and install it before you install NumPy.

After you have confirmed that Python is installed, use the following [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) command to download and install NumPy: 

```bash
sudo pip install numpy
```

If you prefer, you can use a package manager to download and install NumPy:

```bash
sudo apt-get install python-numpy # for debian
sudo yum install python-numpy # for redhat
```

Now install the MXNet package for Python. There are several ways to do this:

* **Perform a system-wide installation** - This option installs MXNet for all users of the system, and requires root permission. It's good for installing MXNet for multiple users.

* **Perform a current-user installation** - This option installs MXNet for the current user only. It's good for installing MXNet for a single user.

* **Set a Python path environment variable** - This option sets a Python environment variable that tells Python where to find the MXNet library. This option is good for developers who are planning to change the MXNet source code.

* **Copy the MXNet source code to the applications directory** - This option is good for users who are planning to do a lot of distributed training.

##### Perform a System-wide Installation
A system-wide installation makes MXNet available for all users of the system where MXNet is installed. Setting this up requires root permission and the Python ```distutils``` module. 
```Distutils``` is often part of the core Python package, and might already be installed on your system. If it isn't, you can install it using your package manager:

```bash
sudo apt-get install python-setuptools # for debian
```

After you have confirmed that ```distutils``` is installed on your system, use the ```cd``` command to navigate to the python folder in the mxnet source directory and run the MXNet Python setup command:

```bash
cd python; sudo python setup.py install
```
##### Current-User Installation
A current-user installation makes MXNet available for only the current user. To install MXNet Python for a single user, use the ``cd`` command to navigate to the python folder in the mxnet source directory, and run the setup command:

```bash
cd python; python setup.py develop --user
```

##### Set a Python Path Environment Variable
Setting a Python path environment variable tells Python where to find the MXNet library. We recommend it for developers who might change the MXNet code itself. The changes are immediately reflected after you pull the code from GitHub, or modify code locally and rebuild the project. You don't need to set up MXNet again after building your changes.

To set the environment variable ```PYTHONPATH``` to point to the MXNet Source Code, add the following line to your ```~/.bashrc``` file:

```bash
export PYTHONPATH=<MXNet source code path>/mxnet/python
```

For example, if you cloned the MXNet source code to your home directory ```~```, you would add the line:

```bash
export PYTHONPATH=~/mxnet/python
```

##### Copying the MXNet Shared Library to the Application Directory

Copying the MXNet Python package into the directory that contains your MXNet application program allows you to perform distributed training without changing your system configurations and settings. Setting this up requires that you also copy the built MXNet library from [Step 1. Build the Shared Library](#build-the-shared-library) to the application directory.
To copy the MXNet Python package and the MXNet shared library into the working directory, run the following commands:

```bash
cp -r ~/mxnet/python/mxnet <path to your working directory>
cp ~/mxnet/lib/libmxnet.so <path to your working directory>/mxnet/
```

For example, if you are in the working directory for the MXNet application, and ```mxnet``` is cloned on the home directory ```~```, you would run the following commands to copy the MXNet python package and the MXNet shared library built in Step 1 from the MXNet C++ Source Code to the MXNet application directory:

```bash
cp -r ~/mxnet/python/mxnet .
cp ~/mxnet/lib/libmxnet.so mxnet/
```

#### Installing the MXNet Package for R
MXNet for R is available for both CPUs and GPUs.

##### Installing MXNet on a Computer with a CPU Processor

To install MXNet on a computer with a CPU processor, choose from two options:

* Use the prebuilt binary package

* Build the library from source code

###### Building MXNet with the Prebuilt Binary Package
For Windows and OS X (Mac) users, MXNet provides a prebuilt binary package for CPUs. The prebuilt package is updated weekly. You can install the package directly in the R console using the following commands:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
```

###### Building MXNet from Source Code

Before you build MXNet for R from source code, complete [Step 1. Build the Shared Library](#build-the-shared-library): Build the shared library from the MXNet C++ source code. After you build the shared library, run the following commands to install the MXNet dependencies and build the MXNet R package:

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

##### Installing MXNet on a Computer with a GPU Processor

To install MXNet on a computer with a GPU processsor, you need the following:

* Microsoft Visual Studio 2013

* The NVidia CUDA Toolkit

* The MXNet package

* CuDNN (to provide a Deep Neural Network library)

To install the required dependencies and install MXNet for R:

1. If [Microsoft Visual Studio 2013](https://www.visualstudio.com/downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). The CUDA Toolkit depends on Visual Studio. To check whether your GPU is compatible with the CUDA Toolkit and for information on installing it, see NVidia's [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/). 
3. Download the MXNet package as a .zip file from the [MXNet Github repository](https://github.com/dmlc/mxnet/) and unpack it. You will be editing the ```"/mxnet/R-package"``` folder.
4. Download the most recent GPU-enabled MXNet package from the [Releases](https://github.com/dmlc/mxnet/releases) tab. Unzip this file and navigate to the ```/nocudnn``` folder. 
**Note:** You will copy some of these extracted files into MXNet's R-package folder. We are now working two folders, 	```R-package/``` and ```nocudnn/```.
5. Download and install [CuDNN V3](https://developer.nvidia.com/cudnn). To get access to the download link, register as an NVIDIA community user. Unpack the .zip file. You will see three folders: ```/bin```, ```/include```, and ```/lib```. Copy these folders into ```nocudnn/3rdparty/cudnn/```, replacing the folders that are already there. You can also unpack the .zip file directly into the nocudnn/ folder.
6. Create a folder called ```R-package/inst/libs/x64```. MXNet supports only 64-bit operating systems, so you need the x64 folder.
7. Copy the following shared libraries (.dll files) into the ```R-package/inst/libs/x64``` folder:
    * nocudnn/lib/libmxnet.dll.
    * The *.dll files in all four subfolders of the nocudnn/3rdparty/ directory. The cudnn and openblas .dll files are in the /bin folders.
You should now have 11 .dll files in the R-package/inst/libs/x64 folder.
8. Copy the ```nocudnn/include/``` folder into ```R-package/inst/```. You should now have a folder called ```R-package/inst/include/``` with three subfolders.
9. Make sure that R is added to your ```PATH``` in the environment variables. Running the where ```R``` command at the command prompt should return the location.
10.	Run ```R CMD INSTALL --no-multiarch R-package```.
 
**Note:** To maximize its portability, the MXNet library is built with the Rcpp end. Computers running Windows need [MSVC](https://en.wikipedia.org/wiki/Visual_C%2B%2B) (Microsoft Visual C++) to handle CUDA toolchain compatibilities.

#### Installing the MXNet Package for Julia 

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

#### Installing the MXNet Package for Scala
There are four ways to install the MXNet package for Scala:

* Use the prebuilt binary package

* Build the library from source code

* Run MXNet on Docker

* Build the dependent libraries from source code

##### Use the Prebuilt Binary Package
For Linux and OS X (Mac) users, MXNet provides prebuilt binary packages that support computers with either GPU or CPU processors. To download and build these packages using ```Maven```, change the ```artifactId``` in the following Maven dependency to match your architecture:

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_<system architecture></artifactId>
  <version>0.1.1</version>
</dependency>
```

For example, to download and build the 64-bit CPU-only version for OS X, use:

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_2.10-linux-x86_64-gpu</artifactId>
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

##### Build the Library from Source Code
Before you build MXNet for Scala from source code, you must complete [Step 1. Build the Shared Library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Scala package:

```bash
make scalapkg
```

This command creates the JAR files for the assembly, core, and example modules. It also creates the native library in the ```native/{your-architecture}/target directory```, which you can use to cooperate with the core module.

To install the MXNet Scala package into your local Maven repository, run the following command from the MXNet source root directory:

```bash
make scalainstall
```

##### Run MXNet on Docker
[Docker](http://docker.com/) is a system that lets you build self-contained versions of a Linux operating system that can run in isolation on your computer. On the self-contained version of Linux, you can run MXNet and other software packages without them interacting with the packages on your computer.


MXNet provides two Docker images for running MXNet:

1. MXNet Docker (CPU) - [https://hub.docker.com/r/kaixhin/mxnet/](https://hub.docker.com/r/kaixhin/mxnet/)

2. MXNet Docker (GPU) - [https://hub.docker.com/r/kaixhin/cuda-mxnet/](https://hub.docker.com/r/kaixhin/cuda-mxnet/)

These Docker images are updated weekly with the latest builds of MXNet. 
For CUDA support, you need the [NVIDIA Docker image](https://github.com/NVIDIA/nvidia-docker).

To run MXNet on Docker:

1. Install Docker on your computer. For more information, see the [Docker documentation](https://docs.docker.com/engine/installation/).
2. Run the command to start the MXNet Docker container.

	For CPU containers, run this command:

	```bash
		sudo docker run -it kaixhin/mxnet
	```

	For GPU containers, run this command:

	```bash
		sudo nvidia-docker run -it kaixhin/cuda-mxnet:7.0
	```

For more details on how to use the MXNet Docker images, see the [MXNet Docker GitHub documentation](https://github.com/Kaixhin/dockerfiles).

##### Build the Dependent Libraries from Source Code
This section provides instructions on how to build MXNet's dependent libraries from source code. This approach is useful in two specific situations:

- If you are using an earlier version of Linux on your server and required packages are either missing or Yum or apt-get didn't install a later version of the packages.

- If you do not have root permission to install packages. In this case, you need to change the installation directory from /usr/local to one where you do have permission. The following examples use the directory ${HOME}.

###### Building GCC from Source Code
To build the GNU Complier Collection (GCC) from source code, you need the 32-bit libc library.

1. Install libc with one of the following system-specific commands:
 
	```bash
		sudo apt-get install libc6-dev-i386 # In Ubuntu
		sudo yum install glibc-devel.i686   # In RHEL (Red Hat Linux)
		sudo yum install glibc-devel.i386   # In CentOS 5.8
		sudo yum install glibc-devel.i686   # In CentOS 6/7
	```
2. To download and extract the GCC source code, run the following commands:

	```bash
		wget http://mirrors.concertpass.com/gcc/releases/gcc-4.8.5/gcc-4.8.5.tar.gz
		tar -zxf gcc-4.8.5.tar.gz
		cd gcc-4.8.5
		./contrib/download_prerequisites
	```
3. To build GCC, run the following commands:

	```bash
		mkdir release && cd release
		../configure --prefix=/usr/local --enable-languages=c,c++
		make -j10
		sudo make install
	```
4. Add the lib path to your ~/.bashrc file:

	```bash
		export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
	```
###### Build OpenCV from Source Code
To build OpenCV from source code, you need the ```cmake``` library .

* If you don't have cmake or if your version of cmake is earlier than 3.6.1 (such as the default version of cmake on RHEL), run the following commands to install a newer version of cmake:

	```bash
		wget https://cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.tar.gz
		tar -zxvf cmake-3.6.1-Linux-x86_64.tar.gz
		alias cmake="cmake-3.6.1-Linux-x86_64/bin/cmake"
	```

* To download and extract the OpenCV source code, run the following commands:

	```bash
		wget https://codeload.github.com/opencv/opencv/zip/2.4.13
		unzip 2.4.13
		cd opencv-2.4.13
		mkdir release
		cd release/
	```

* Build OpenCV. The following commands build OpenCV without GPU support, which might significantly slow down an MXNet program running on a GPU processor. It also disables 1394 which might generate a warning:
	
	```bash
		cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
		make -j8
		sudo make install
	```
* Add the lib path to your ```~/.bashrc``` file:

	```bash
		export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
	```
# Common Installation Problems
This section provides solutions for common installation problems.
## Mac OS X Error Message
**Message:** link error ld: library not found for -lgomp

**Cause:** The GNU implementation of OpenMP is not in the operating system's library path.

**Solution:** Add the location of OpenMP to the library path:
 
* Create the locate database by running the following command:

	```bash
		sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.locate.plist
	```
* Locate the OpenMP library by running the following command:

	```bash
		locate libgomp.dylib
	```
* To add OpenMP to your library path, replace ```path1``` in the following command with the output from the last command:

	```bash
		ln -s path1 /usr/local/lib/libgomp.dylib
	```

* To build your changes, run the following command:

	```bash
		make -j$(sysctl -n hw.ncpu)
	```
## R Error Message
**Message** Unable to load mxnet after enabling CUDA

**Solution:** If you enabled CUDA when you installed MXNet, but can't load mxnet, add the following lines to your ```$RHOME/etc/ldpaths``` environment variable:
 
```bash
export CUDA_HOME=/usr/local/cuda 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

**Note:** To find your $RHOME environment variable, use the ```R.home()``` command in R.

# Next Steps
* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
