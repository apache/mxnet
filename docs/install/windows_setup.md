# Installing MXNet in Windows

On Windows, you can download and install the prebuilt MXNet package, or download, build, and install MXNet yourself.

## Build the Shared Library
You can either use a prebuilt binary package or build from source to build the MXNet shared library - ```libmxnet.dll```.

### Installing the Prebuilt Package on Windows
MXNet provides a prebuilt package for Windows. The prebuilt package includes the MXNet library, all of the dependent third-party libraries, a sample C++ solution for Visual Studio, and the Python installation script. To install the prebuilt package:

1. Download the latest prebuilt package from the [Releases](https://github.com/dmlc/mxnet/releases) tab of MXNet.
2. Unpack the package into a folder, with an appropriate name, such as ```D:\MXNet```.
3. Open the folder, and install the package by double-clicking ```setupenv.cmd```. This sets up all of the environment variables required by MXNet.
4. Test the installation by opening the provided sample C++ Visual Studio solution and building it.


&nbsp;
This produces a library called ```libmxnet.dll```.

### Building and Installing Packages on Windows

To build and install MXNet yourself, you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2015](https://www.visualstudio.com/downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and Install [CMake](https://cmake.org/) if it is not already installed.
3. Download and install [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download).
4. Unzip the OpenCV package.
5. Set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (```c:\utils\opencv\build``` for example).
6. If you have Intel Math Kernel Library (MKL) installed, set ```MKL_ROOT``` to point to ```MKL``` directory that contains the ```include``` and ```lib```. Typically, you can find the directory in
```C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\mkl```.
7. If you don't have the Intel Math Kernel Library (MKL) installed, download and install [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/).
8. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories. Typically, you can find the directory in ```C:\Program files (x86)\OpenBLAS\```.
9. Download and install [CuDNN](https://developer.nvidia.com/cudnn). To get access to the download link, register as an NVIDIA community user.

After you have installed all of the required dependencies, build the MXNet source code:

1. Download the MXNet source code from [GitHub](https://github.com/dmlc/mxnet). Don't forget to pull the submodules:
```
    git clone https://github.com/apache/incubator-mxnet.git ~/mxnet --recursive
```
2. Start a Visual Studio command prompt.
3. Use [CMake](https://cmake.org/) to create a Visual Studio solution in ```./build``` or some other directory. Make sure to specify the architecture in the 
[CMake](https://cmake.org/) command:
```
    mkdir build
    cd build
    cmake -G "Visual Studio 14 Win64" ..
```
4. In Visual Studio, open the solution file,```.sln```, and compile it.
These commands produce a library called ```mxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.



&nbsp;
Next, we install ```graphviz``` library that we use for visualizing network graphs you build on MXNet. We will also install [Jupyter Notebook](http://jupyter.readthedocs.io/)  used for running MXNet tutorials and examples.
- Install ```graphviz``` by downloading MSI installer from [Graphviz Download Page](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).
**Note** Make sure to add graphviz executable path to PATH environment variable. Refer [here for more details](http://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)
- Install ```Jupyter``` by installing [Anaconda for Python 2.7](https://www.continuum.io/downloads)
**Note** Do not install Anaconda for Python 3.5. MXNet has few compatibility issue with Python 3.5.

&nbsp;

We have installed MXNet core library. Next, we will install MXNet interface package for programming language of your choice:
- [Python](#install-the-mxnet-package-for-python)
- [R](#install-mxnet-for-r)
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

#### Installing MXNet with the Prebuilt Binary Package
For Windows users, MXNet provides prebuilt binary packages.
You can install the package directly in the R console.

For CPU-only package:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
  options(repos = cran)
  install.packages("mxnet")
```

For GPU-enabled package:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU"
  options(repos = cran)
  install.packages("mxnet")
```

#### Building MXNet from Source Code

Run the following commands to install the MXNet dependencies and build the MXNet R package.

```r
  Rscript -e "install.packages('devtools', repo = 'https://cloud.r-project.org/')"
```

```bash
  cd R-package
  Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cloud.r-project.org/')); install_deps(dependencies = TRUE)"
  cd ..
  make rpkg
```

**Note:** R-package is a folder in the MXNet source.

These commands create the MXNet R package as a tar.gz file that you can install as an R package. To install the R package, run the following command, use your MXNet version number:

```bash
  R CMD INSTALL mxnet_current_r.tar.gz
```

### Installing MXNet on a Computer with a GPU Processor

To install MXNet R package on a computer with a GPU processor, you need the following:

* Microsoft Visual Studio 2013

* The NVidia CUDA Toolkit

* The MXNet package

* CuDNN (to provide a Deep Neural Network library)

To install the required dependencies and install MXNet for R:

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). The CUDA Toolkit depends on Visual Studio. To check whether your GPU is compatible with the CUDA Toolkit and for information on installing it, see NVidia's [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).
3. Clone the MXNet github repo.

```sh
git clone --recursive https://github.com/dmlc/mxnet
```

The `--recursive` is to clone all the submodules used by MXNet. You will be editing the ```"/mxnet/R-package"``` folder.
4. Download prebuilt GPU-enabled MXNet libraries for Windows from https://github.com/yajiedesign/mxnet/releases. You will need `mxnet_x64_vc14_gpu.7z` and `prebuildbase_win10_x64_vc14.7z`.
5. Download and install [CuDNN](https://developer.nvidia.com/cudnn).
6. Create a folder called ```R-package/inst/libs/x64```. MXNet supports only 64-bit operating systems, so you need the x64 folder.
7. Copy the following shared libraries (.dll files) into the ```R-package/inst/libs/x64``` folder:
```
cublas64_80.dll
cudart64_80.dll
cudnn64_5.dll
curand64_80.dll
libgcc_s_seh-1.dll
libgfortran-3.dll
libmxnet.dll
libmxnet.lib
libopenblas.dll
libquadmath-0.dll
nvrtc64_80.dll
```
These dlls can be found in `prebuildbase_win10_x64_vc14/3rdparty/cudart`, `prebuildbase_win10_x64_vc14/3rdparty/openblas/bin`, `mxnet_x64_vc14_gpu/build`, `mxnet_x64_vc14_gpu/lib` and the `cuDNN` downloaded from NVIDIA.
8. Copy the header files from `dmlc`, `mxnet` and `nnvm` into `./R-package/inst/include`. It should look like:

```
./R-package/inst
└── include
    ├── dmlc
    ├── mxnet
    └── nnvm
```
9. Make sure that R is added to your ```PATH``` in the environment variables. Running the ```where R``` command at the command prompt should return the location.
10. Now open the Windows CMD and change the directory to the `mxnet` folder. Then use the following commands
to build R package:

```bat
echo import(Rcpp) > R-package\NAMESPACE
echo import(methods) >> R-package\NAMESPACE
Rscript -e "install.packages('devtools', repos = 'https://cloud.r-project.org')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cloud.r-project.org')); install_deps(dependencies = TRUE)"
cd ..

R CMD INSTALL --no-multiarch R-package

Rscript -e "require(mxnet); mxnet:::mxnet.export('R-package')"
rm R-package/NAMESPACE
Rscript -e "require(devtools); install_version('roxygen2', version = '5.0.1', repos = 'https://cloud.r-project.org/', quiet = TRUE)"
Rscript -e "require(roxygen2); roxygen2::roxygenise('R-package')"

R CMD INSTALL --build --no-multiarch R-package
```

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
* [How To](http://mxnet.io/faq/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
