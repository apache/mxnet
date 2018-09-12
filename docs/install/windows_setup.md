# Installing MXNet in Windows

We provide two options to build and install MXNet yourself using [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) or [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/).

You can also install MXNet with the following programming language support:
- [Python](#install-the-mxnet-package-for-python)
- [R](#install-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)


**Option 1: Build with Visual Studio 2017**

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

```
cd C:\
git clone https://github.com/apache/incubator-mxnet.git --recursive
```

3. Follow [this link](https://docs.microsoft.com/en-us/visualstudio/install/modify-visual-studio) to modify ```Individual components```, and check ```VC++ 2017 version 15.4 v14.11 toolset```, and click ```Modify```.
4. Change the version of the Visual studio 2017 to v14.11 using the following command (by default the VS2017 is installed in the following path):

```
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.11
```

5. Create a build dir using the following command and go to the directory, for example:

```
mkdir C:\build
cd C:\build
```

6. CMake the MXNet source code by using following command:

```
cmake -G "Visual Studio 15 2017 Win64" -T cuda=9.1,host=x64 -DUSE_CUDA=1 -DUSE_CUDNN=1 -DUSE_NVRTC=1 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_LIST=Common -DCUDA_TOOLSET=9.1 -DCUDNN_INCLUDE=C:\cuda\include -DCUDNN_LIBRARY=C:\cuda\lib\x64\cudnn.lib "C:\incubator-mxnet"
```

NOTE: make sure the DCUDNN_INCLUDE and DCUDNN_LIBRARY pointing to the “include” and “cudnn.lib” of your CUDA installed location, and the ```C:\incubator-mxnet``` is the location of the source code you just git in the previous step

7. After the CMake successfully completed, compile the the MXNet source code by using following command:

```
msbuild mxnet.sln /p:Configuration=Release;Platform=x64 /maxcpucount
```

**Option 2: Build with Visual Studio 2015**

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
Next, we install ```graphviz``` library that we use for visualizing network graphs you build on MXNet. We will also install [Jupyter Notebook](http://jupyter.readthedocs.io/)  used for running MXNet tutorials and examples.
- Install ```graphviz``` by downloading MSI installer from [Graphviz Download Page](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).
**Note** Make sure to add graphviz executable path to PATH environment variable. Refer [here for more details](http://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)
- Install ```Jupyter``` by installing [Anaconda for Python 2.7](https://www.anaconda.com/download/)
**Note** Do not install Anaconda for Python 3.5. MXNet has a few compatibility issues with Python 3.5.

We have installed MXNet core library. Next, we will install MXNet interface package for programming language of your choice:
- [Python](#install-the-mxnet-package-for-python)
- [R](#install-mxnet-package-for-r)
- [Julia](#install-the-mxnet-package-for-julia)
- **Scala** is not yet available for Windows

## Install MXNet for Python

1. Install ```Python``` using windows installer available [here](https://www.python.org/downloads/release/python-2712/).
2. Install ```Numpy``` using windows installer available [here](https://scipy.org/index.html).
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

## Install the MXNet Package for R
MXNet for R is available for both CPUs and GPUs.

### Installing MXNet-R on a Computer with a CPU Processor

To install MXNet on a computer with a CPU processor, choose from two options:

* Use the prebuilt binary package
* Build the library from source code

#### Installing MXNet-R with the Prebuilt Binary Package(CPU)
For Windows users, MXNet provides prebuilt binary packages.
You can install the package directly in the R console.

For CPU-only package:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
  options(repos = cran)
  install.packages("mxnet")
```

#### Building MXNet-R from Source Code(CPU)
1. Clone the MXNet github repo.

```sh
git clone --recursive https://github.com/apache/incubator-mxnet
```

The `--recursive` is to clone all the submodules used by MXNet. You will be editing the ```"/mxnet/R-package"``` folder.

2. Download prebuilt GPU-enabled MXNet libraries for Windows from [Windows release](https://github.com/yajiedesign/mxnet/releases). You will need `mxnet_x64_vc14_cpu.7z` and `prebuildbase_win10_x64_vc14.7z` where X stands for your CUDA toolkit version

3. Create a folder called ```R-package/inst/libs/x64```. MXNet supports only 64-bit operating systems, so you need the x64 folder.

4. Copy the following shared libraries (.dll files) into the ```R-package/inst/libs/x64``` folder:
```
libgcc_s_seh-1.dll
libgfortran-3.dll
libmxnet.dll
libmxnet.lib
libopenblas.dll
libquadmath-0.dll
mxnet.dll
unzip.exe
unzip32.dll
vcomp140.dll
wget.exe
```
These dlls can be found in `prebuildbase_win10_x64_vc14/3rdparty`, `mxnet_x64_vc14_cpu/build`, `mxnet_x64_vc14_cpu/lib`.

5. Copy the header files from `dmlc`, `mxnet`, `mxshadow` and `nnvm` from mxnet_x64_vc14_cpu/include and mxnet_x64_vc14_cpu/nvnm/include into `./R-package/inst/include`. It should look like:

```
./R-package/inst
└── include
    ├── dmlc
    ├── mxnet
    ├── mshadow
    └── nnvm

```
6. Make sure that R executable is added to your ```PATH``` in the environment variables. Running the ```where R``` command at the command prompt should return the location.
7. Also make sure that Rtools is installed and the executable is added to your ```PATH``` in the environment variables.
8. Temporary patch - im2rec currently results in crashes during the build. Remove the im2rec.h and im2rec.cc files in R-package/src/ from cloned repository and comment out the two im2rec lines in [R-package/src/mxnet.cc](https://github.com/apache/incubator-mxnet/blob/master/R-package/src/mxnet.cc) as shown below.
```bat
#include "./kvstore.h"
#include "./export.h"
//#include "./im2rec.h"
......
......
  DataIterCreateFunction::InitRcppModule();
  KVStore::InitRcppModule();
  Exporter::InitRcppModule();
//  IM2REC::InitRcppModule();
}

```

9. Now open the Windows CMD with admin rights and change the directory to the `mxnet` folder(cloned repository). Then use the following commands
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


### Installing MXNet-R on a Computer with a GPU Processor
To install MXNet on a computer with a GPU processor, choose from two options:

* Use the prebuilt binary package
* Build the library from source code

However, a few dependencies remain for both options.  You will need the following:
* Install [Nvidia-drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us) if not installed. Latest driver based on your system configuration is recommended.

* Install [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/) (VS2015 or VS2017 is required by CUDA)

* Install  [NVidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)(cu92 is recommended though we support cu80, cu90, cu91 and cu92)

* Download and install [CuDNN](https://developer.nvidia.com/cudnn) (to provide a Deep Neural Network library). Latest version recommended.

Note: A pre-requisite to above softwares is [Nvidia-drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us) which we assume is installed.

#### Installing MXNet-R with the Prebuilt Binary Package(GPU)
For Windows users, MXNet provides prebuilt binary packages.
You can install the package directly in the R console after you have the above software installed.

For GPU package:

```r
  cran <- getOption("repos")
  cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
  options(repos = cran)
  install.packages("mxnet")
```
Change cu92 to cu80, cu90 or cu91 based on your CUDA toolkit version. Currently, MXNet supports these versions of CUDA.

#### Building MXNet-R from Source Code(GPU)
After you have installed above software, continue with the following steps to build MXNet-R:
1. Clone the MXNet github repo.

```sh
git clone --recursive https://github.com/apache/incubator-mxnet
```

The `--recursive` is to clone all the submodules used by MXNet. You will be editing the ```"/mxnet/R-package"``` folder.

2. Download prebuilt GPU-enabled MXNet libraries for Windows from https://github.com/yajiedesign/mxnet/releases. You will need `mxnet_x64_vc14_gpu_cuX.7z` and `prebuildbase_win10_x64_vc14.7z` where X stands for your CUDA toolkit version

3. Create a folder called ```R-package/inst/libs/x64```. MXNet supports only 64-bit operating systems, so you need the x64 folder.

4. Copy the following shared libraries (.dll files) into the ```R-package/inst/libs/x64``` folder:
```
libgcc_s_seh-1.dll
libgfortran-3.dll
libmxnet.dll
libmxnet.lib
libopenblas.dll
libquadmath-0.dll
mxnet.dll
unzip.exe
unzip32.dll
vcomp140.dll
wget.exe
```
These dlls can be found in `prebuildbase_win10_x64_vc14/3rdparty`, `mxnet_x64_vc14_gpu_cuX/build`, `mxnet_x64_vc14_gpu_cuX/lib`.

5. Copy the header files from `dmlc`, `mxnet`, `mxshadow` and `nnvm` from mxnet_x64_vc14_gpuX/include and mxnet_x64_vc14_gpuX/nvnm/include into `./R-package/inst/include`. It should look like:

```
./R-package/inst
└── include
    ├── dmlc
    ├── mxnet
    ├── mshadow
    └── nnvm

```
6. Make sure that R executable is added to your ```PATH``` in the environment variables. Running the ```where R``` command at the command prompt should return the location.
7. Also make sure that Rtools is installed and the executable is added to your ```PATH``` in the environment variables.
8. Temporary patch - im2rec currently results in crashes during the build. Remove the im2rec.h and im2rec.cc files in R-package/src/ from cloned repository and comment out the two im2rec lines in [R-package/src/mxnet.cc](https://github.com/apache/incubator-mxnet/blob/master/R-package/src/mxnet.cc) as shown below.
```bat
#include "./kvstore.h"
#include "./export.h"
//#include "./im2rec.h"
......
......
  DataIterCreateFunction::InitRcppModule();
  KVStore::InitRcppModule();
  Exporter::InitRcppModule();
//  IM2REC::InitRcppModule();
}

```
9. Now open the Windows CMD with admin rights and change the directory to the `mxnet` folder(cloned repository). Then use the following commands
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

MXNet-Scala is not yet available for Windows.

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/faq/index.html)
* [Architecture](http://mxnet.io/architecture/index.html)
