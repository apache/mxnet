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

# Install MXNet with MKL-DNN

A better training and inference performance is expected to be achieved on Intel-Architecture CPUs with MXNet built with [Intel MKL-DNN](https://github.com/intel/mkl-dnn) on multiple operating system, including Linux, Windows and MacOS.
In the following sections, you will find build instructions for MXNet with Intel MKL-DNN on Linux, MacOS and Windows.

Please find MKL-DNN optimized operators and other features in the [MKL-DNN operator list](https://github.com/apache/incubator-mxnet/blob/v1.5.x/docs/tutorials/mkldnn/operator_list.md).

The detailed performance data collected on Intel Xeon CPU with MXNet built with Intel MKL-DNN can be found [here](https://mxnet.apache.org/api/faq/perf#intel-cpu).


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

To achieve better performance, the Intel OpenMP and llvm OpenMP are recommended as below instruction. Otherwise, default GNU OpenMP will be used and you may get the sub-optimal performance. If you don't have the full [MKL](https://software.intel.com/en-us/intel-mkl) library installation, you might use OpenBLAS as the blas library, by setting USE_BLAS=openblas.

```
# build with llvm OpenMP and Intel MKL/openblas
mkdir build && cd build
cmake -DUSE_CUDA=OFF -DUSE_MKL_IF_AVAILABLE=ON -DUSE_MKLDNN=ON -DUSE_OPENMP=ON -DUSE_OPENCV=ON ..
make -j $(nproc)
```

```
# build with Intel MKL and Intel OpenMP
make -j $(nproc) USE_OPENCV=1 USE_MKLDNN=1 USE_BLAS=mkl USE_INTEL_PATH=/opt/intel
```

```
# build with openblas and GNU OpenMP(sub-optimal performance)
make -j $(nproc) USE_OPENCV=1 USE_MKLDNN=1 USE_BLAS=openblas
```

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
LIBRARY_PATH=$(brew --prefix llvm)/lib/ make -j $(sysctl -n hw.ncpu) CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ USE_OPENCV=1 USE_OPENMP=1 USE_MKLDNN=1 USE_BLAS=apple
```

<h2 id="3">Windows</h2>

On Windows, you can use [Micrsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) and [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) to compile MXNet with Intel MKL-DNN.
[Micrsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is recommended.

**Visual Studio 2015**

To build and install MXNet yourself, you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and Install [CMake 3](https://cmake.org/files/v3.14/cmake-3.14.0-win64-x64.msi) if it is not already installed.
3. Download [OpenCV 3](https://sourceforge.net/projects/opencvlibrary/files/3.4.5/opencv-3.4.5-vc14_vc15.exe/download), and unzip the OpenCV package, set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (e.g.,```OpenCV_DIR = C:\opencv\build ```). Also, add the OpenCV bin directory (```C:\opencv\build\x64\vc14\bin``` for example) to the ``PATH`` variable.
4. If you have Intel Math Kernel Library (Intel MKL) installed, set ```MKL_ROOT``` to point to ```MKL``` directory that contains the ```include``` and ```lib```. If you want to use MKL blas, you should set ```-DUSE_BLAS=mkl``` when cmake. Typically, you can find the directory in ```C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl```.
5. If you don't have the Intel Math Kernel Library (MKL) installed, download and install [OpenBLAS](http://sourceforge.net/projects/openblas/files/v0.2.14/), or build the latest version of OpenBLAS from source. Note that you should also download ```mingw64.dll.zip``` along with openBLAS and add them to PATH.
6. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories. Typically, you can find the directory in ```C:\Downloads\OpenBLAS\```.

After you have installed all of the required dependencies, build the MXNet source code:

1. Start a Visual Studio command prompt by click windows Start menu>>Visual Studio 2015>>VS2015 X64 Native Tools Command Prompt, and download the MXNet source code from [GitHub](https://github.com/apache/incubator-mxnet) by the command:
```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd C:\incubator-mxent
```
2. Enable Intel MKL-DNN by -DUSE_MKLDNN=1. Use [CMake 3](https://cmake.org/) to create a Visual Studio solution in ```./build```. Make sure to specify the architecture in the
command:
```
>mkdir build
>cd build
>cmake -G "Visual Studio 14 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release
```
3. Enable Intel MKL-DNN and Intel MKL as BLAS library by the command:
```
>"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\bin\mklvars.bat" intel64
>cmake -G "Visual Studio 14 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=mkl -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl"
```
4. After the CMake successfully completed, in Visual Studio, open the solution file ```.sln``` and compile it, or compile the MXNet source code by using following command:
```r
msbuild mxnet.sln /p:Configuration=Release;Platform=x64 /maxcpucount
```
   These commands produce mxnet library called ```libmxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder. Also ```libmkldnn.dll``` with be in the ```./build/3rdparty/mkldnn/src/Release/```

5. Make sure that all the dll files used above(such as `libmkldnn.dll`, `libmklml*.dll`, `libiomp5.dll`, `libopenblas*.dll`, etc) are added to the system PATH. For convinence, you can put all of them to ```\windows\system32```. Or you will come across `Not Found Dependencies` when loading MXNet.

**Visual Studio 2017**

User can follow the same steps of Visual Studio 2015 to build MXNET with MKL-DNN, but change the version related command, for example,```C:\opencv\build\x64\vc15\bin``` and build command is as below:

```
>cmake -G "Visual Studio 15 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=mkl -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release -DMKL_ROOT="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl"

```

<h2 id="4">Verify MXNet with python</h2>

Preinstall python and some dependent modules:
```
pip install numpy graphviz
set PYTHONPATH=[workdir]\incubator-mxnet\python
```
or install mxnet
```
cd python
sudo python setup.py install
python -c "import mxnet as mx;print((mx.nd.ones((2, 3))*2).asnumpy());"
```
Expected Output:
```
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
dnnl_verbose,info,DNNL v1.1.2 (commit cb2cc7ac17ff4e2ef50805c7048d33256d82be4d)
dnnl_verbose,info,Detected ISA is Intel AVX-512 with Intel DL Boost
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd16b:f0,,,32x32x256x256,7.43701
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd16b16a:f0,,,32x32x3x3,0.202148
dnnl_verbose,exec,cpu,convolution,jit:avx512_common,forward_inference,src_f32::blocked:aBcd16b:f0 wei_f32::blocked:ABcd16b16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,,alg:convolution_direct,mb32_ic32oc32_ih256oh256kh3sh1dh0ph1_iw256ow256kw3sw1dw0pw1,20.7539
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd16b16a:f0,,,32x32x3x3,1.86694
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,32x32x256x256,35.9771
```

You can find step-by-step guidance to do profiling for MKLDNN primitives in [Profiling MKLDNN Operators](https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/profiler.html#Profiling-MKLDNN-Operators).

<h2 id="5">Enable MKL BLAS</h2>

With MKL BLAS, the performace is expected to furtherly improved with variable range depending on the computation load of the models.
You can redistribute not only dynamic libraries but also headers, examples and static libraries on accepting the license [Intel Simplified license](https://software.intel.com/en-us/license/intel-simplified-software-license).
Installing the full MKL installation enables MKL support for all operators under the linalg namespace.

  1. Download and install the latest full MKL version following instructions on the [intel website.](https://software.intel.com/en-us/mkl) You can also install MKL through [YUM](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-yum-repo) or [APT](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo) Repository.

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
MKL_VERBOSE Intel(R) MKL 2019.0 Update 3 Product build 20190125 for Intel(R) 64 architecture Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) enabled processors, Lnx 2.40GHz lp64 intel_thread NMICDev:0
MKL_VERBOSE SGEMM(T,N,12,10,8,0x7f7f927b1378,0x1bc2140,8,0x1ba8040,8,0x7f7f927b1380,0x7f7f7400a280,12) 8.93ms CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:40 WDiv:HOST:+0.000
```

<h2 id="6">Enable graph optimization</h2>

Graph optimization with subgraph is available and enabled by default in master branch. For MXNet release v1.5, you can manually enable it by:

```
export MXNET_SUBGRAPH_BACKEND=MKLDNN
```

This limitations of this experimental feature are:

- Use this feature only for inference. When training, be sure to turn the feature off by unsetting the `MXNET_SUBGRAPH_BACKEND` environment variable.

- This feature will only run on the CPU, even if you're using a GPU-enabled build of MXNet.


<h2 id="7">Quantization and Inference with INT8</h2>

Benefiting from Intel MKL-DNN, MXNet built with Intel MKL-DNN brings outstanding performance improvement on quantization and inference with INT8 Intel CPU Platform on Intel Xeon Scalable Platform.

- [CNN Quantization Examples](https://github.com/apache/incubator-mxnet/tree/master/example/quantization).

- [Model Quantization for Production-Level Neural Network Inference](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN).

<h2 id="8">Next Steps and Support</h2>

- For questions or support specific to MKL, visit the [Intel MKL](https://software.intel.com/en-us/mkl) website.

- For questions or support specific to MKL, visit the [Intel MKLDNN](https://github.com/intel/mkl-dnn) website.

- If you find bugs, please open an issue on GitHub for [MXNet with MKL](https://github.com/apache/incubator-mxnet/labels/MKL) or [MXNet with MKLDNN](https://github.com/apache/incubator-mxnet/labels/MKLDNN).
