# 在 Windows 上安装 MXNet

在 Windows 上，你可以下载安装预编译 MXNet 包，或者自己下载，编译，然后安装。

## 编译共享库
你可以使用预编译二进制包或者通过源码自己编译共享库 -  ```libmxnet.dll```。

### 在 Windows 上安装预编译包
MXNet 为 Windows 提供了一个预编译包。这个包里包含了 MXNet 库，所有的第三方依赖库，一个 Visual Studio 的 C++ 解决方案例子，和一个 Python 安装脚本。安装预编译：

1. 从 MXNet [Releases](https://github.com/dmlc/mxnet/releases) 表里下载最新的预编译包。这有两个版本。一个 GPU 版本(需要 CUDA 和 CUDNN v03),一个非 GPU 版本。请根据你的硬件配置选择合适的版本。哪个版本适应哪种硬件配置，更详细的信息参考 [Requirements for GPU](./setup_zh.md#使用-gpu-的要求)。
2. 将预编译包解压到文件夹中，可以给文件夹取个合适的名字，比如 ```D:\MXNet```。
3. 打开文件夹，双击 ```setupenv.cmd``` 进行安装。它会配置好所有 MXNet 需要的所有环境变量。
4. 可以通过里面提供的  Visual Studio C++ 解决方案来测试安装是否成功。

&nbsp;

这里生成了一个库名字是 ```libmxnet.dll```。

### 在 Windows 上编译然后安装

如果要编译并安装 MXNet 你需要下面的依赖，安装依赖：
1. 如果没有安装 [Microsoft Visual Studio 2013](https://www.visualstudio.com/downloads/) ，下载并安装，可以下载免费的社区版。
2. 安装 [Visual C++ Compiler Nov 2013 CTP](https://www.microsoft.com/en-us/download/details.aspx?id=41151)。
3. 备份 ```C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC``` 目录下的所有文件到其他地方。
4. 拷贝 ```C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP``` 目录（或者其他你解压zip文件的目录）下的所有文件到 ```C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC``` 目录下，并且覆盖所有文件。
5. 下载并安装 [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download)。
6. 解压 OpenCV 包。
7. 配置环境变量 ```OpenCV_DIR``` 指向 ```OpenCV build directory```。
8. 如果你没有安装 Intel Math Kernel Library (MKL)。请下载并安装 [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/)。
9. 配置环境变量 ```OpenBLAS_HOME``` 指向 ```OpenBLAS``` 目录，它包含了 ```include``` 和 ```lib```。通常你可以在 ```C:\Program files (x86)\OpenBLAS\``` 下找到这个目录。
10. 下载并安装 [CuDNN](https://developer.nvidia.com/cudnn)。你需要注册 NVIDIA community 用户才能拿到下载链接。

安装完所有的以来后，编译 MXNet 源码：

1. 从 [GitHub](https://github.com/dmlc/mxnet) 上下载源码。
2. 用 [CMake](https://cmake.org/) 在 ```./build``` 下生成一个 Visual Studio 解决方案。
3. 用 Visual Studio 打开解决方案 ```.sln```，并且编译。
这些命令会在 ```./build/Release/``` 或者 ```./build/Debug``` 目录下生成一个库名字叫 ```mxnet.dll``` 


&nbsp;


接下来，安装 ```graphviz``` 库，通过它，我们可以用可视化网络图( visualizing network graphs)来构建 MXNet。我还将安装 [Jupyter Notebook](http://jupyter.readthedocs.io/) 用来运行 MXNet 教程和例子。
- 在 [Graphviz 下载页](http://www.graphviz.org/Download_windows.php) 下载 MSI 安装包，然后安装。
**注意** 确保 graphviz 的执行路径在 ```PATH``` 环境变量中。参考 [这里获取更详细的信息](http://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft) 
- 通过 [Anaconda for Python 2.7](https://www.continuum.io/downloads) 安装 ```Jupyter``` 
**注意** 不要安装 Anaconda for Python 3.5 。 MXNet 在 Python 3.5 上有一部分兼容性问题。

&nbsp;

我们已经安装完了 MXNet 的核心库(core library)，接下来我们要选择一种编程语言，安装语言接口包:
- [Python](#安装-python-版-mxnet)
- [R](#安装-r-版-mxnet)
- [Julia](#安装-julia-版-mxnet)
- [Scala](#安装-scala-版-mxnet)


## 安装 Python 版 MXNet

1. 在 [这里](https://www.python.org/downloads/release/python-2712/) 找到可用的 Windows 安装包，安装 ```Python``` 。
2. 在 [这里](http://scipy.org/install.html) 找到可用的 Windows 安装包，安装 ```Numpy``` 。
3. 接下来，我们为 MXNet 安装 Python 接口。你可以在 [MXNet on GitHub](https://github.com/dmlc/mxnet/tree/master/python/mxnet) 找到 Python 接口包。

```bash
    # Assuming you are in root mxnet source code folder
    cd python
    sudo python setup.py install
```

完成！我们已经安装好了 MXNet 的 Python 接口。运行下面的命令来检车安装是否成功。

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

我们实际上是用 MXNet 做了一个小的张量(tensor)计算！到这里你已经在电脑上完全配置好了 MXNet。

## 安装 R 版 MXNet
R 版 MXNet 同时支持 CPU 和 GPU.

### 安装使用 CPU 的 MXNet

从下面的两个选项里选择一个，来安装仅使用 CPU 的 MXNet

* 使用预编译二进制包
* 使用源码编译

#### 使用预编译二进制包
对于 Windows 用户，提供了一个 CPU 版本的 MXNet 包。这个包每周更新，你可以在 R 控制台里 直接输入下面的命令来安装：

```r
  install.packages("drat", repos="https://cran.rstudio.com")
  drat:::addRepo("dmlc")
  install.packages("mxnet")
```

#### 使用源码编译

运行下面的命令来安装，编译 MXNet R 语言包的依赖：
BRun the following commands to install the MXNet dependencies and build the MXNet R package.

```r
  Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
```
```bash
  cd R-package
  Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
  cd ..
  make rpkg
```

**注意:** R-package 是 MXNet 源码的一个目录。

这些命令可以生成一个 tar.gz 格式的 MXNet R 语言包，运行下面命令来安装这个语言包：

```bash
  R CMD INSTALL mxnet_current_r.tar.gz
```

### 安装使用 GPU 的 MXNet

安装使用 GPU 的 MXNet 你需要如下步骤：

* Microsoft Visual Studio 2013

* NVidia CUDA 工具包

* MXNet 包

* CuDNN (提供一个深度神经网络库)

安装依赖和 R 版本 MXNet：

1. 如果没有安装 [Microsoft Visual Studio 2013](https://www.visualstudio.com/downloads/) ，下载并安装，可以下载免费的社区版。
2. 安装 [CUDA 工具包](https://developer.nvidia.com/cuda-toolkit)。CUDA 工具包依赖 Visual Studio。检查你的 GPU 和 CUDA 工具包的兼容性。更详细的安装信息参考 [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)。
3. 以 zip 格式从 [MXNet Github 仓库](https://github.com/dmlc/mxnet/) 下载 MXNet 包，解压，你将要编辑 ```"/mxnet/R-package"``` 目录。
4. 从 [Releases](https://github.com/dmlc/mxnet/releases) 表中下载最新的 GPU-enabled MXNet 包，解压并导航到 ```/nocudnn``` 文件夹。
**注意:** 你将会拷贝这些额外的文件到 MXNet 的 R-package文件夹中。目前我们只在两个目录下工作 ```R-package/``` 和 ```nocudnn/```。
5. 下载并安装 [CuDNN](https://developer.nvidia.com/cudnn)。你需要注册 NVIDIA community 用户才能拿到下载链接。解压这个 ```.zip``` 文件，你会发现这些文件夹： ```/bin```, ```/include```, 和 ```/lib```。复制这些文件夹到 ```nocudnn/3rdparty/cudnn/```，替换原有文件。你也可以解压这个 ```.zip``` 文件到 ```nocudnn/``` 目录中。
6. 创建文件夹 ```R-package/inst/libs/x64```，MXNet只支持64位操作系统，因此需要 x64 文件夹。
7. 拷贝下面的共享库(.dll 文件)到 ```R-package/inst/libs/x64``` 目录中：
    * nocudnn/lib/libmxnet.dll.
    * ```nocudnn/3rdparty/``` 的四个子目录下的所有 *.dll 文件。cudnn 和 openblas .dll 在 ```/bin``` 目录下。
在 ```R-package/inst/libs/x64``` 目录下，现在应该有11个 .dll 文件。
8. 拷贝 ```nocudnn/include/``` 目录到  ```R-package/inst/``` 中。现在应该有一个文件夹叫 ```R-package/inst/include/```，它有三个子目录。
9. 确认 R 已经添加到你的 ```PATH``` 环境变量中。在命令提示符窗口运行```where R``` 命令，可以返回具体位置。
10. 运行 ```R CMD INSTALL --no-multiarch R-package```。

**注意:** 为了最大化可移植性，MXNet 库是使用 Rcpp 编译的。Windows 电脑需要 [MSVC](https://en.wikipedia.org/wiki/Visual_C%2B%2B) (Microsoft Visual C++) 来处理 CUDA 工具链的兼容性。

## 安装 Julia 版 MXNet
## (此方法是适用于 Linux 和 OS X 的，原来的英文文档可能存在错误，我只是翻译在这里，并不保证可以运行,如果谁更新了方法，可以联系我修改中文翻译hebeilijianzhang@163.com)

MXNet Julia 语言包托管在一个单独的仓库 ```MXNet.jl```。地址是 [GitHub](https://github.com/dmlc/MXNet.jl)。Julia需要与已经安装的 libmxnet 绑定。使用下面的命令来配置 ```MXNET_HOME``` 环境变量：

```bash
export MXNET_HOME=/<path to>/libmxnet
```

这里的路径(path to)指的是已经安装的 libmxnet 的根目录。也即是说你可以在 ```$MXNET_HOME/lib``` 目录下找到 ```libmxnet.so``` 文件。举例来讲 libmxnet 的根目录是 ```~``` 你应该执行如下命令：

```bash
	export MXNET_HOME=/~/libmxnet
```

你也许想把这条命令添加到 ```~/.bashrc``` 文件中，如果是的话，你可以在 Julia 控制台中执行如下命令来安装 Julia 语言包。

```julia
	Pkg.add("MXNet")
```

MXNet Julia 语言包更详细的安装教程可以参考 [MXNet Julia documentation](http://dmlc.ml/MXNet.jl/latest/user-guide/install/).

## 安装 Scala 版 MXNet
## (此方法是适用于 Linux 和 OS X 的，原来的英文文档可能存在错误，我只是翻译在这里，并不保证可以运行,如果谁更新了方法，可以联系我修改中文翻译hebeilijianzhang@163.com)

有两种方式安装 MXNet Scala 语言包：

* 使用预编译的二进制包

* 通过源码编译

### 使用预编译的二进制包
对于 Linux 和 OS X (Mac) 用户，MXNet 提供了编译好的二进制包，同时支持 CPU 和 GPU。可以通过 ```Maven``` 来下载使用这个包，根据你的需求修改下面 Maven 依赖里的 ```artifactId``` ：

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_<system architecture></artifactId>
  <version>0.1.1</version>
</dependency>
```

比如，下载 Linux 上64位 CPU-only 版本:

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_2.10-linux-x86_64-cpu</artifactId>
  <version>0.1.1</version>
</dependency>
```

如果你的本地环境和集成包的有微小区别，比如，你使用了 openblas 包而不是 atlas 包，你最好使用 mxnet-core 包并且将编译好的本地Java库放到你的加载目录中去：

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-core_2.10</artifactId>
  <version>0.1.1</version>
</dependency>
```

### 通过源码编译
在编译 MXNet Scala 语言包前，你必须已经完成了 [编译共享库](#编译共享库)。然后在 MXNet 源码根目录运行下面命令来编译 Scala 语言包：

```bash
  make scalapkg
```

这条命令会生成一个JAR文件，里面包含了封装(assembly)，核心(core)和例子(example)。同时还会在 ```native/{your-architecture}/target directory``` 下生成一个本地库，你可以通过它配合 core 使用。

在 MXNet 根目录下运行下面的命令，可以讲 MXNet Scala 语言包安装到你本地的 Maven 仓库

```bash
  make scalainstall
```

# 下一步
* [教程](http://mxnet.io/tutorials/index.html)
* [如何使用](http://mxnet.io/how_to/index.html)
* [架构设计](http://mxnet.io/architecture/index.html)