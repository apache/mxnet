<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Build/Install MXNet with MKL-DNN

A better training and inference performance is expected to be achieved on Intel-Architecture CPUs with MXNet built with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) on multiple operating system, including Linux, Windows and MacOS.
In the following sections, you will find build instructions for MXNet with Intel MKL-DNN on Linux, MacOS and Windows.

The detailed performance data collected on Intel Xeon CPU with MXNet built with Intel MKL-DNN can be found [here](https://mxnet.incubator.apache.org/faq/perf.html#intel-cpu).


<h2 id="0">Contents</h2>

* [1. Linux](#1)
* [2. MacOS](#2)
* [3. Windows](#3)
* [4. Verify MXNet with python](#4)
* [5. Enable MKL BLAS](#5)
* [6. Enable graph optimization](#6)
* [7. Quantization](#7)
* [8. Support](#8)

<h2 id="1">Linux</h2>

### Prerequisites

```
sudo apt-get update
sudo apt-get install -y build-essential git
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libopencv-dev
sudo apt-get install -y graphviz
```

### Clone MXNet sources

```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
```

### Build MXNet with MKL-DNN

```
make -j $(nproc) USE_OPENCV=1 USE_MKLDNN=1 USE_BLAS=mkl USE_INTEL_PATH=/opt/intel
```

If you don't have the full [MKL](https://software.intel.com/en-us/intel-mkl) library installation, you might use OpenBLAS as the blas library, by setting USE_BLAS=openblas.

<h2 id="2">MacOS</h2>

### Prerequisites

Install the dependencies, required for MXNet, with the following commands:

- [Homebrew](https://brew.sh/)
- llvm (clang in macOS does not support OpenMP)
- OpenCV (for computer vision operations)

```
# Paste this command in Mac terminal to install Homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# install dependency
brew update
brew install pkg-config
brew install graphviz
brew tap homebrew/core
brew install opencv
brew tap homebrew/versions
brew install llvm
```

### Clone MXNet sources

```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
```

### Build MXNet with MKL-DNN

```
LIBRARY_PATH=$(brew --prefix llvm)/lib/ make -j $(sysctl -n hw.ncpu) CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ USE_OPENCV=1 USE_OPENMP=1 USE_MKLDNN=1 USE_BLAS=apple USE_PROFILER=1
```

<h2 id="3">Windows</h2>

On Windows, you can use [Micrsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) and [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) to compile MXNet with Intel MKL-DNN.
[Micrsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is recommended.

**Visual Studio 2015**

To build and install MXNet yourself, you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and Install [CMake 3](https://cmake.org/) if it is not already installed.
3. Download and install [OpenCV 3](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download).
4. Unzip the OpenCV package.
5. Set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (```C:\opencv\build\x64\vc14``` for example). Also, you need to add the OpenCV bin directory (```C:\opencv\build\x64\vc14\bin``` for example) to the ``PATH`` variable.
6. If you have Intel Math Kernel Library (MKL) installed, set ```MKL_ROOT``` to point to ```MKL``` directory that contains the ```include``` and ```lib```. If you want to use MKL blas, you should set ```-DUSE_BLAS=mkl``` when cmake. Typically, you can find the directory in
```C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\mkl```.
7. If you don't have the Intel Math Kernel Library (MKL) installed, download and install [OpenBLAS](http://sourceforge.net/projects/openblas/files/v0.2.14/). Note that you should also download ```mingw64.dll.zip`` along with openBLAS and add them to PATH.
8. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories. Typically, you can find the directory in ```C:\Program files (x86)\OpenBLAS\```. 

After you have installed all of the required dependencies, build the MXNet source code:

1. Download the MXNet source code from [GitHub](https://github.com/apache/incubator-mxnet). Don't forget to pull the submodules:
```
git clone --recursive https://github.com/apache/incubator-mxnet.git
```

2. Copy file `3rdparty/mkldnn/config_template.vcxproj` to incubator-mxnet root.

3. Start a Visual Studio command prompt.

4. Use [CMake 3](https://cmake.org/) to create a Visual Studio solution in ```./build``` or some other directory. Make sure to specify the architecture in the 
[CMake 3](https://cmake.org/) command:
```
mkdir build
cd build
cmake -G "Visual Studio 14 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release
```

5. In Visual Studio, open the solution file,```.sln```, and compile it.
These commands produce a library called ```libmxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder.
Also ```libmkldnn.dll``` with be in the ```./build/3rdparty/mkldnn/src/Release/```

6. Make sure that all the dll files used above(such as `libmkldnn.dll`, `libmklml.dll`, `libiomp5.dll`, `libopenblas.dll`, etc) are added to the system PATH. For convinence, you can put all of them to ```\windows\system32```. Or you will come across `Not Found Dependencies` when loading MXNet.

**Visual Studio 2017**

To build and install MXNet yourself using [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/), you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and install [CMake 3](https://cmake.org/files/v3.11/cmake-3.11.0-rc4-win64-x64.msi) if it is not already installed.
3. Download and install [OpenCV](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.1/opencv-3.4.1-vc14_vc15.exe/download).
4. Unzip the OpenCV package.
5. Set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (e.g., ```OpenCV_DIR = C:\utils\opencv\build```).
6. If you don’t have the Intel Math Kernel Library (MKL) installed, download and install [OpenBlas](https://sourceforge.net/projects/openblas/files/v0.2.20/OpenBLAS%200.2.20%20version.zip/download).
7. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories (e.g., ```OpenBLAS_HOME = C:\utils\OpenBLAS```).

After you have installed all of the required dependencies, build the MXNet source code:

1. Start ```cmd``` in windows.

2. Download the MXNet source code from GitHub by using following command:

```r
cd C:\
git clone --recursive https://github.com/apache/incubator-mxnet.git
```

3. Copy file `3rdparty/mkldnn/config_template.vcxproj` to incubator-mxnet root.

4. Follow [this link](https://docs.microsoft.com/en-us/visualstudio/install/modify-visual-studio) to modify ```Individual components```, and check ```VC++ 2017 version 15.4 v14.11 toolset```, and click ```Modify```.

5. Change the version of the Visual studio 2017 to v14.11 using the following command (by default the VS2017 is installed in the following path):

```r
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.11
```

6. Create a build dir using the following command and go to the directory, for example:

```r
mkdir C:\build
cd C:\build
```

7. CMake the MXNet source code by using following command:

```r
cmake -G "Visual Studio 15 2017 Win64" .. -T host=x64 -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release
```

8. After the CMake successfully completed, compile the the MXNet source code by using following command:

```r
msbuild mxnet.sln /p:Configuration=Release;Platform=x64 /maxcpucount
```

9. Make sure that all the dll files used above(such as `libmkldnn.dll`, `libmklml.dll`, `libiomp5.dll`, `libopenblas.dll`, etc) are added to the system PATH. For convinence, you can put all of them to ```\windows\system32```. Or you will come across `Not Found Dependencies` when loading MXNet.

<h2 id="4">Verify MXNet with python</h2>

```
cd python
sudo python setup.py install
python -c "import mxnet as mx;print((mx.nd.ones((2, 3))*2).asnumpy());"

Expected Output:

[[ 2.  2.  2.]
 [ 2.  2.  2.]]
```

### Verify whether MKL-DNN works

After MXNet is installed, you can verify if MKL-DNN backend works well with a single Convolution layer.

```
import mxnet as mx
import numpy as np

num_filter = 32
kernel = (3, 3)
pad = (1, 1)
shape = (32, 32, 256, 256)

x = mx.sym.Variable('x')
w = mx.sym.Variable('w')
y = mx.sym.Convolution(data=x, weight=w, num_filter=num_filter, kernel=kernel, no_bias=True, pad=pad)
exe = y.simple_bind(mx.cpu(), x=shape)

exe.arg_arrays[0][:] = np.random.normal(size=exe.arg_arrays[0].shape)
exe.arg_arrays[1][:] = np.random.normal(size=exe.arg_arrays[1].shape)

exe.forward(is_train=False)
o = exe.outputs[0]
t = o.asnumpy()
```

More detailed debugging and profiling information can be logged by setting the environment variable 'MKLDNN_VERBOSE':
```
export MKLDNN_VERBOSE=1
```
For example, by running above code snippet, the following debugging logs providing more insights on MKL-DNN primitives `convolution` and `reorder`. That includes: Memory layout, infer shape and the time cost of primitive execution.
```
mkldnn_verbose,exec,reorder,jit:uni,undef,in:f32_nchw out:f32_nChw16c,num:1,32x32x256x256,6.47681
mkldnn_verbose,exec,reorder,jit:uni,undef,in:f32_oihw out:f32_OIhw16i16o,num:1,32x32x3x3,0.0429688
mkldnn_verbose,exec,convolution,jit:avx512_common,forward_inference,fsrc:nChw16c fwei:OIhw16i16o fbia:undef fdst:nChw16c,alg:convolution_direct,mb32_g1ic32oc32_ih256oh256kh3sh1dh0ph1_iw256ow256kw3sw1dw0pw1,9.98193
mkldnn_verbose,exec,reorder,jit:uni,undef,in:f32_oihw out:f32_OIhw16i16o,num:1,32x32x3x3,0.0510254
mkldnn_verbose,exec,reorder,jit:uni,undef,in:f32_nChw16c out:f32_nchw,num:1,32x32x256x256,20.4819
```

<h2 id="5">Enable MKL BLAS</h2>

With MKL BLAS, the performace is expected to furtherly improved with variable range depending on the computation load of the models.
You can redistribute not only dynamic libraries but also headers, examples and static libraries on accepting the license [Intel® Simplified license](https://software.intel.com/en-us/license/intel-simplified-software-license).
Installing the full MKL installation enables MKL support for all operators under the linalg namespace.

  1. Download and install the latest full MKL version following instructions on the [intel website.](https://software.intel.com/en-us/mkl)

  2. Run `make -j ${nproc} USE_BLAS=mkl`

  3. Navigate into the python directory

  4. Run `sudo python setup.py install`

### Verify whether MKL works

After MXNet is installed, you can verify if MKL BLAS works well with a single dot layer.

```
import mxnet as mx
import numpy as np

shape_x = (1, 10, 8)
shape_w = (1, 12, 8)

x_npy = np.random.normal(0, 1, shape_x)
w_npy = np.random.normal(0, 1, shape_w)

x = mx.sym.Variable('x')
w = mx.sym.Variable('w')
y = mx.sym.batch_dot(x, w, transpose_b=True)
exe = y.simple_bind(mx.cpu(), x=x_npy.shape, w=w_npy.shape)

exe.forward(is_train=False)
o = exe.outputs[0]
t = o.asnumpy()
```

You can open the `MKL_VERBOSE` flag by setting environment variable:
```
export MKL_VERBOSE=1
```
Then by running above code snippet, you probably will get the following output message which means `SGEMM` primitive from MKL are called. Layout information and primitive execution performance are also demonstrated in the log message.
```
Numpy + Intel(R) MKL: THREADING LAYER: (null)
Numpy + Intel(R) MKL: setting Intel(R) MKL to use INTEL OpenMP runtime
Numpy + Intel(R) MKL: preloading libiomp5.so runtime
MKL_VERBOSE Intel(R) MKL 2018.0 Update 1 Product build 20171007 for Intel(R) 64 architecture Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) enabled processors, Lnx 2.40GHz lp64 intel_thread NMICDev:0
MKL_VERBOSE SGEMM(T,N,12,10,8,0x7f7f927b1378,0x1bc2140,8,0x1ba8040,8,0x7f7f927b1380,0x7f7f7400a280,12) 8.93ms CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:40 WDiv:HOST:+0.000
```

<h2 id="6">Enable graph optimization</h2>

Graph optimization by subgraph feature are available in master branch. You can build from source and then use below command to enable this *experimental* feature for better performance:

```
export MXNET_SUBGRAPH_BACKEND=MKLDNN
```

This limitations of this experimental feature are:

- Use this feature only for inference. When training, be sure to turn the feature off by unsetting the `MXNET_SUBGRAPH_BACKEND` environment variable.

- This feature will only run on the CPU, even if you're using a GPU-enabled build of MXNet. 

- [MXNet Graph Optimization and Quantization Technical Information and Performance Details](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN).

<h2 id="7">Quantization and Inference with INT8</h2>

Benefiting from Intel® MKL-DNN, MXNet built with Intel® MKL-DNN brings outstanding performance improvement on quantization and inference with INT8 Intel® CPU Platform on Intel® Xeon® Scalable Platform.

- [CNN Quantization Examples](https://github.com/apache/incubator-mxnet/tree/master/example/quantization).

<h2 id="8">Next Steps and Support</h2>

- For questions or support specific to MKL, visit the [Intel MKL](https://software.intel.com/en-us/mkl) website.

- For questions or support specific to MKL, visit the [Intel MKLDNN](https://github.com/intel/mkl-dnn) website.

- If you find bugs, please open an issue on GitHub for [MXNet with MKL](https://github.com/apache/incubator-mxnet/labels/MKL) or [MXNet with MKLDNN](https://github.com/apache/incubator-mxnet/labels/MKLDNN).
