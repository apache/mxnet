# Installing MXNet in Windows

On Windows, you can download and install the prebuilt MXNet package, or download, build, and install MXNet yourself.

## Build the Shared Library
You can either use a prebuilt binary package or build from source to build the MXNet shared library - ```libmxnet.dll```.

### Installing the Prebuilt Package on Windows
MXNet provides a prebuilt package for Windows. The prebuilt package includes the MXNet library, all of the dependent third-party libraries, a sample C++ solution for Visual Studio, and the Python installation script. To install the prebuilt package:

1. Download the latest prebuilt package from the [Releases](https://github.com/dmlc/mxnet/releases) tab of MXNet.
   There are two versions. One with GPU support (using CUDA and CUDNN v3), and one without GPU support. Choose the version that suits your hardware configuration. For more information on which version works on each hardware configuration, see [Requirements for GPU](http://mxnet.io/get_started/setup.html#requirements-for-using-gpus).
2. Unpack the package into a folder, with an appropriate name, such as ```D:\MXNet```.
3. Open the folder, and install the package by double-clicking ```setupenv.cmd```. This sets up all of the environment variables required by MXNet.
4. Test the installation by opening the provided sample C++ Visual Studio solution and building it.


&nbsp;
This produces a library called ```libmxnet.dll```.

### Building and Installing Packages on Windows

To build and install MXNet yourself, you need the following dependencies. Install the required dependencies:

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
These commands produce a library called ```mxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.



&nbsp;
Next, we install ```graphviz``` library that we use for visualizing network graphs you build on MXNet. We will also install [Jupyter Notebook](http://jupyter.readthedocs.io/)  used for running MXNet tutorials and examples.
- Install ```graphviz``` by downloading MSI installer from [Graphviz Download Page](http://www.graphviz.org/Download_windows.php).
**Note** Make sure to add graphviz executable path to PATH environment variable. Refer [here for more details](http://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)
- Install ```Jupyter``` by installing [Anaconda for Python 2.7](https://www.continuum.io/downloads)
**Note** Do not install Anaconda for Python 3.5. MXNet has few compatibility issue with Python 3.5.

&nbsp;

We have installed MXNet core library. Next, we will install MXNet interface package for programming language of your choice:
- [Python](#install-the-mxnet-package-for-python)
- [R](#install-the-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- [Scala](#install-the-mxnet-package-for-scala)

## Install MXNet for Python

1. Install ```Python``` using windows installer available [here](https://www.python.org/downloads/release/python-2712/).
2. Install ```Numpy``` using windows installer available [here](http://scipy.org/install.html).
3. Next, we install Python package interface for MXNet. You can find the Python interface package for [MXNet on GitHub](https://github.com/dmlc/mxnet/tree/master/python/mxnet).

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
We actually did a small tensor computation using MXNet! You are all set with MXNet on your Windows machine.

## Install MXNet for R
MXNet for R is available for both CPUs and GPUs.

### Installing MXNet on a Computer with a CPU Processor

To install MXNet on a computer with a CPU processor, choose from two options:

* Use the prebuilt binary package
* Build the library from source code

#### Building MXNet with the Prebuilt Binary Package
For Windows users, MXNet provides a prebuilt binary package for CPUs. The prebuilt package is updated weekly. You can install the package directly in the R console using the following commands:

```r
  install.packages("drat", repos="https://cran.rstudio.com")
  drat:::addRepo("dmlc")
  install.packages("mxnet")
```

#### Building MXNet from Source Code

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

### Installing MXNet on a Computer with a GPU Processor

To install MXNet on a computer with a GPU processor, you need the following:

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
9. Make sure that R is added to your ```PATH``` in the environment variables. Running the ```where R``` command at the command prompt should return the location.
10.	Run ```R CMD INSTALL --no-multiarch R-package```.

**Note:** To maximize its portability, the MXNet library is built with the Rcpp end. Computers running Windows need [MSVC](https://en.wikipedia.org/wiki/Visual_C%2B%2B) (Microsoft Visual C++) to handle CUDA toolchain compatibilities.

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

## Installing the MXNet Package for Scala
There are four ways to install the MXNet package for Scala:

* Use the prebuilt binary package

* Build the library from source code

### Use the Prebuilt Binary Package
For Linux and OS X (Mac) users, MXNet provides prebuilt binary packages that support computers with either GPU or CPU processors. To download and build these packages using ```Maven```, change the ```artifactId``` in the following Maven dependency to match your architecture:

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
Before you build MXNet for Scala from source code, you must complete [Step 1. Build the Shared Library](#build-the-shared-library). After you build the shared library, run the following command from the MXNet source root directory to build the MXNet Scala package:

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
