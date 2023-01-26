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

# Install MXNet with oneDNN

A better training and inference performance is expected to be achieved on Intel-Architecture CPUs with MXNet built with [oneDNN](https://github.com/oneapi-src/oneDNN) on multiple operating system, including Linux, Windows and MacOS.
In the following sections, you will find build instructions for MXNet with oneDNN on Linux, MacOS and Windows.

The detailed performance data collected on Intel Xeon CPU with MXNet built with oneDNN can be found [here](https://mxnet.apache.org/api/faq/perf#intel-cpu).


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
git clone --recursive https://github.com/apache/mxnet.git
cd mxnet
```

### Build MXNet with oneDNN

To achieve better performance, the Intel OpenMP and llvm OpenMP are recommended as below instruction. Otherwise, default GNU OpenMP will be used and you may get the sub-optimal performance. If you don't have the full [MKL](https://software.intel.com/en-us/intel-mkl) library installation, you might use OpenBLAS as the blas library, by setting USE_BLAS=Open.

```
# build with llvm OpenMP and Intel MKL/OpenBlas
mkdir build && cd build
cmake -DUSE_CUDA=OFF -DUSE_ONEDNN=ON -DUSE_OPENMP=ON -DUSE_OPENCV=ON ..
make -j $(nproc)
```

```
# build with Intel MKL and Intel OpenMP
mkdir build && cd build
cmake -DUSE_CUDA=OFF -DUSE_ONEDNN=ON -DUSE_BLAS=mkl ..
make -j $(nproc)
```

```
# build with openblas and GNU OpenMP (sub-optimal performance)
mkdir build && cd build
cmake -DUSE_CUDA=OFF -DUSE_ONEDNN=ON -DUSE_BLAS=Open ..
make -j $(nproc)
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
git clone --recursive https://github.com/apache/mxnet.git
cd mxnet
```

### Build MXNet with oneDNN

```
LIBRARY_PATH=$(brew --prefix llvm)/lib/ make -j $(sysctl -n hw.ncpu) CC=$(brew --prefix llvm)/bin/clang CXX=$(brew --prefix llvm)/bin/clang++ USE_OPENCV=1 USE_OPENMP=1 USE_ONEDNN=1 USE_BLAS=apple
```

<h2 id="3">Windows</h2>

On Windows, you can use [Micrsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) and [Microsoft Visual Studio 2017](https://www.visualstudio.com/downloads/) to compile MXNet with oneDNN.
[Micrsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is recommended.

**Visual Studio 2015**

To build and install MXNet yourself, you need the following dependencies. Install the required dependencies:

1. If [Microsoft Visual Studio 2015](https://www.visualstudio.com/vs/older-downloads/) is not already installed, download and install it. You can download and install the free community edition.
2. Download and Install [CMake 3](https://cmake.org/files/v3.14/cmake-3.14.0-win64-x64.msi) if it is not already installed.
3. Download [OpenCV 3](https://sourceforge.net/projects/opencvlibrary/files/3.4.5/opencv-3.4.5-vc14_vc15.exe/download), and unzip the OpenCV package, set the environment variable ```OpenCV_DIR``` to point to the ```OpenCV build directory``` (e.g.,```OpenCV_DIR = C:\opencv\build ```). Also, add the OpenCV bin directory (```C:\opencv\build\x64\vc14\bin``` for example) to the ``PATH`` variable.
4. If you have Intel Math Kernel Library (Intel MKL) installed, set ```MKLROOT``` environment variable to point to ```MKL``` directory that contains the ```include``` and ```lib```. If you want to use MKL blas, you should set ```-DUSE_BLAS=mkl``` when cmake. Typically, you can find the directory in ```C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl```.
5. If you don't have the Intel Math Kernel Library (MKL) installed, download and install [OpenBLAS](http://sourceforge.net/projects/openblas/files/v0.2.14/), or build the latest version of OpenBLAS from source. Note that you should also download ```mingw64.dll.zip``` along with openBLAS and add them to PATH.
6. Set the environment variable ```OpenBLAS_HOME``` to point to the ```OpenBLAS``` directory that contains the ```include``` and ```lib``` directories. Typically, you can find the directory in ```C:\Downloads\OpenBLAS\```.

After you have installed all of the required dependencies, build the MXNet source code:

1. Start a Visual Studio command prompt by click windows Start menu>>Visual Studio 2015>>VS2015 X64 Native Tools Command Prompt, and download the MXNet source code from [GitHub](https://github.com/apache/mxnet) by the command:
```
git clone --recursive https://github.com/apache/mxnet.git
cd C:\mxent
```
2. Enable oneDNN by -DUSE_ONEDNN=1. Use [CMake 3](https://cmake.org/) to create a Visual Studio solution in ```./build```. Make sure to specify the architecture in the
command:
```
>mkdir build
>cd build
>cmake -G "Visual Studio 14 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=Open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_ONEDNN=1 -DCMAKE_BUILD_TYPE=Release
```
3. Enable oneDNN and Intel MKL as BLAS library by the command:
```
>"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\bin\mklvars.bat" intel64
>cmake -G "Visual Studio 14 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=mkl -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_ONEDNN=1 -DCMAKE_BUILD_TYPE=Release
```
4. After the CMake successfully completed, in Visual Studio, open the solution file ```.sln``` and compile it, or compile the MXNet source code by using following command:
```r
msbuild mxnet.sln /p:Configuration=Release;Platform=x64 /maxcpucount
```
   These commands produce mxnet library called ```libmxnet.dll``` in the ```./build/Release/``` or ```./build/Debug``` folder. Also ```libmkldnn.dll``` with be in the ```./build/3rdparty/onednn/src/Release/```

5. Make sure that all the dll files used above(such as `libmkldnn.dll`, `libmklml*.dll`, `libiomp5.dll`, `libopenblas*.dll`, etc) are added to the system PATH. For convinence, you can put all of them to ```\windows\system32```. Or you will come across `Not Found Dependencies` when loading MXNet.

**Visual Studio 2017**

User can follow the same steps of Visual Studio 2015 to build MXNET with oneDNN, but change the version related command, for example,```C:\opencv\build\x64\vc15\bin``` and build command is as below:

```
>cmake -G "Visual Studio 15 Win64" .. -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_NVRTC=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=mkl -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_NAME=All -DUSE_ONEDNN=1 -DCMAKE_BUILD_TYPE=Release

```

<h2 id="4">Verify MXNet with python</h2>

Preinstall python and some dependent modules:
```
pip install numpy graphviz
set PYTHONPATH=[workdir]\mxnet\python
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
### Verify whether oneDNN works

After MXNet is installed, you can verify if oneDNN backend works well with a single Convolution layer.
```
from mxnet import np
from mxnet.gluon import nn

num_filter = 32
kernel = (3, 3)
pad = (1, 1)
shape = (32, 32, 256, 256)

conv_layer = nn.Conv2D(channels=num_filter, kernel_size=kernel, padding=pad)
conv_layer.initialize()

data = np.random.normal(size=shape)
o = conv_layer(data)
print(o)
```

More detailed debugging and profiling information can be logged by setting the environment variable 'DNNL_VERBOSE':
```
export DNNL_VERBOSE=1
```
For example, by running above code snippet, the following debugging logs providing more insights on oneDNN primitives `convolution` and `reorder`. That includes: Memory layout, infer shape and the time cost of primitive execution.
```
dnnl_verbose,info,oneDNN v2.3.2 (commit e2d45252ae9c3e91671339579e3c0f0061f81d49)
dnnl_verbose,info,cpu,runtime:OpenMP
dnnl_verbose,info,cpu,isa:Intel AVX-512 with Intel DL Boost
dnnl_verbose,info,gpu,runtime:none
dnnl_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,,,32x32x256x256,8.34912
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb32a:f0,,,32x32x3x3,0.0229492
dnnl_verbose,exec,cpu,convolution,brgconv:avx512_core,forward_inference,src_f32::blocked:acdb:f0 wei_f32::blocked:Acdb32a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:acdb:f0,,alg:convolution_direct,mb32_ic32oc32_ih256oh256kh3sh1dh0ph1_iw256ow256kw3sw1dw0pw1,10.5898
```

You can find step-by-step guidance to do profiling for oneDNN primitives in [Profiling oneDNN Operators](https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/profiler.html#Profiling-MKLDNN-Operators).

<h2 id="5">Enable MKL BLAS</h2>

With MKL BLAS, the performace is expected to furtherly improved with variable range depending on the computation load of the models.
You can redistribute not only dynamic libraries but also headers, examples and static libraries on accepting the license [Intel Simplified license](https://software.intel.com/en-us/license/intel-simplified-software-license).
Installing the full MKL installation enables MKL support for all operators under the linalg namespace.

  1. Download and install the latest full MKL version following instructions on the [intel website.](https://software.intel.com/en-us/mkl) You can also install MKL through [YUM](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/yum-dnf-zypper.html) or [APT](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html) Repository.

  2. Create and navigate to build directory `mkdir build && cd build`

  3. Run `cmake -DUSE_CUDA=OFF -DUSE_BLAS=mkl ..`

  4. Run `make -j`

  5. Navigate into the python directory

  6. Run `sudo python setup.py install`

### Verify whether MKL works

After MXNet is installed, you can verify if MKL BLAS works well with a linear matrix solver.

```
from mxnet import np
coeff = np.array([[7, 0], [5, 2]])
y = np.array([14, 18])
x = np.linalg.solve(coeff, y)
print(x)
```

You can get the verbose log output from mkl library by setting environment variable:
```
export MKL_VERBOSE=1
```
Then by running above code snippet, you should get the similar output to message below (`SGESV` primitive from MKL was executed). Layout information and primitive execution performance are also demonstrated in the log message.
```
mkl-service + Intel(R) MKL: THREADING LAYER: (null)
mkl-service + Intel(R) MKL: setting Intel(R) MKL to use INTEL OpenMP runtime
mkl-service + Intel(R) MKL: preloading libiomp5.so runtime
Intel(R) MKL 2020.0 Update 1 Product build 20200208 for Intel(R) 64 architecture Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) with support of Vector Neural Network Instructions enabled processors, Lnx 2.70GHz lp64 intel_thread
MKL_VERBOSE SGESV(2,1,0x7f74d4002780,2,0x7f74d4002798,0x7f74d4002790,2,0) 77.58us CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:56
```

<h2 id="6">Graph optimization</h2>

To better utilise oneDNN potential, using graph optimizations is recommended. There are few limitations of this feature:

- It works only for inference.
- Only subclasses of HybridBlock and Symbol can call optimize_for API.
- This feature will only run on the CPU, even if you're using a GPU-enabled build of MXNet.

If your use case met above conditions, graph optimizations can be enabled by just simple call `optimize_for` API. Example below:
```
from mxnet import np
from mxnet.gluon import nn

data = np.random.normal(size=(32,3,224,224))

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=64, kernel_size=(3,3)))
net.add(nn.Activation('relu'))
net.initialize()
print("=" * 5, " Not optimized ", "=" * 5)
o = net(data)
print(o)

net.optimize_for(data, backend='ONEDNN')
print("=" * 5, " Optimized ", "=" * 5)
o = net(data)
print(o)

```

Above code snippet should produce similar output to the following one (printed tensors are omitted) :
```
===== Not optimized =====
[15:05:43] ../src/storage/storage.cc:202: Using Pooled (Naive) StorageManager for CPU
dnnl_verbose,info,oneDNN v2.3.2 (commit e2d45252ae9c3e91671339579e3c0f0061f81d49)
dnnl_verbose,info,cpu,runtime:OpenMP
dnnl_verbose,info,cpu,isa:Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions
dnnl_verbose,info,gpu,runtime:none
dnnl_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,,,32x3x224x224,8.87793
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb64a:f0,,,64x3x3x3,0.00708008
dnnl_verbose,exec,cpu,convolution,brgconv:avx512_core,forward_inference,src_f32::blocked:acdb:f0 wei_f32::blocked:Acdb64a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:acdb:f0,,alg:convolution_direct,mb32_ic3oc64_ih224oh222kh3sh1dh0ph0_iw224ow222kw3sw1dw0pw0,91.511
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb64a:f0,,,64x3x3x3,0.00610352
dnnl_verbose,exec,cpu,eltwise,jit:avx512_common,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,32x64x222x222,85.4392
===== Optimized =====
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:Acdb64a:f0 dst_f32::blocked:abcd:f0,,,64x3x3x3,0.00610352
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb64a:f0,,,64x3x3x3,0.00585938
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,,,32x3x224x224,3.98999
dnnl_verbose,exec,cpu,convolution,brgconv:avx512_core,forward_inference,src_f32::blocked:acdb:f0 wei_f32::blocked:Acdb64a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:acdb:f0,attr-post-ops:eltwise_relu:0:1 ,alg:convolution_direct,mb32_ic3oc64_ih224oh222kh3sh1dh0ph0_iw224ow222kw3sw1dw0pw0,20.46
```
After optimization of Convolution + ReLU oneDNN executes both operations within single convolution primitive.

<h2 id="7">Quantization and Inference with INT8</h2>

MXNet built with oneDNN brings outstanding performance improvement on quantization and inference with INT8 Intel CPU Platform on Intel Xeon Scalable Platform.

- [CNN Quantization Examples](https://github.com/apache/mxnet/tree/master/example/quantization).

- [Model Quantization for Production-Level Neural Network Inference](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN).

<h2 id="8">Next Steps and Support</h2>

- For questions or support specific to MKL, visit the [Intel MKL](https://software.intel.com/en-us/mkl) website.

- For questions or support specific to oneDNN, visit the [oneDNN](https://github.com/oneapi-src/oneDNN) website.

- If you find bugs, please open an issue on GitHub for [MXNet with MKL](https://github.com/apache/mxnet/labels/MKL) or [MXNet with oneDNN](https://github.com/apache/mxnet/labels/MKLDNN).
