# 在 Ubuntu 上安装 MXNet
MXNet 目前支持 Python, R, Julia, and Scala。对于在 Ubuntu上使用Python的用户，MXNet 提供了一个安装全部 MXNet 依赖和 MXNet 库的 Git Bash 脚本.

这是个非常简单的安装脚本，可以在 Ubuntu 12 及以上的电脑中安装 Python 和 R 版的MXNet。在 home 目录 ```~/mxnet``` 下找到安装好的 MXNet。


## 为GPU版本安装准备环境

如果使用GPU你需要配置 CUDA 和 cuDNN。

首先,下载并安装 [CUDA 8 工具包](https://developer.nvidia.com/cuda-toolkit).

然后下载 [cudnn 5](https://developer.nvidia.com/cudnn).

解压此文件并且进入 cudnn 根目录。将头文件和库文件移动到你本地的 CUDA 工具包文件夹:

```bash
    tar xvzf cudnn-8.0-linux-x64-v5.1-ga.tgz
    sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    sudo ldconfig
```
最后，将配置添加到 config.mk 文件中:

```bash
    cp make/config.mk .
```

## 快速安装
### 安装 Python 版 MXNet
克隆 MXNet 源码仓库到本地电脑需要使用 ```git```。

```bash
    # Install git if not already installed.
    sudo apt-get update
    sudo apt-get -y install git
```

克隆 MXNet 的源码仓库到你的电脑上，运行安装脚本，然后刷新环境变量。另外，这个脚本还会安装好所有的MXNet的所有依赖: ```Numpy```, ```OpenBLAS``` and ```OpenCV```。

安装这些大概需要花费5分钟。

```bash
    # Clone mxnet repository. In terminal, run the commands WITHOUT "sudo"
    git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive

    # If building with GPU, add configurations to config.mk file:
    cd ~/mxnet
    cp make/config.mk .
    echo "USE_CUDA=1" >>config.mk
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
    echo "USE_CUDNN=1" >>config.mk

    # Install MXNet for Python with all required dependencies
    cd ~/mxnet/setup-utils
    bash install-mxnet-ubuntu-python.sh

    # We have added MXNet Python package path in your ~/.bashrc.
    # Run the following command to refresh environment variables.
    $ source ~/.bashrc
```

点击 [这里](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-ubuntu-python.sh) 可以查看安装脚本的具体内容。

### 安装 R 版 MXNet

MXNet 要求 R 版本 3.2.0 及以上。如果你正在使用低版本的 R， 在运行安装脚本之前，运行下面的命令来升级的你 R 版本。

```bash
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
    sudo add-apt-repository ppa:marutter/rdev

    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install r-base r-base-dev
```

安装 R 版 MXNet:

```bash
    cd ~/mxnet/setup-utils
    bash install-mxnet-ubuntu-r.sh
```

点击 [这里](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-ubuntu-r.sh) 可以查看安装脚本的具体内容。

## 标准安装

安装 MXNet 一共分两步：

1. 将 MXNet C++ 代码编译成共享库。
2. 为 MXNet 安装特定语言包。

**注意:** 可以通过编辑 ```make/config.mk``` 来修改编译选项，然后通过 ```make``` 命令开始编译。

### 编译共享库

在 Ubuntu 13.10 及以上的版本中，你需要如下依赖：

- Git (从 GitHub 上下载代码)

- libatlas-base-dev (线性代数计算)

- libopencv-dev (机器视觉处理)

用下面命令安装依赖库：

```bash
    sudo apt-get update
    sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev
```

安装完依赖后，使用下面命令从 GitHub 上下载 MXNet 源码：

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
如果编译支持 ```GPU``` 的版本，使用如下命令将GPU配置添加到 config.mk 中：

```bash
    echo "USE_CUDA=1" >>config.mk
    echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
    echo "USE_CUDNN=1" >>config.mk
```

然后编译 MXNet:

```bash
    make -j$(nproc)
```

执行完这些命令可以生成一个动态库名字是： ```libmxnet.so```

接下来，安装 ```graphviz``` 库，通过它，我们可以用可视化网络图( visualizing network graphs)来构建 MXNet。我还将安装 [Jupyter Notebook](http://jupyter.readthedocs.io/) 用来运行 MXNet 教程和例子。

```bash
    sudo apt-get install -y python-pip
    sudo pip install graphviz
    sudo pip install Jupyter
```

&nbsp;

我们已经安装完了 MXNet 的核心库(core library)，接下来我们要选择一种编程语言，安装语言接口包:
- [R](#标准安装-r-版-mxnet)
- [Julia](#安装-julia-版-mxnet)
- [Scala](#安装-scala-版-mxnet)

### 标准安装 R 版 MXNet

运行下面命令来安装MXNet依赖并且编译 MXNet R 语言包

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

### 安装 Julia 版 MXNet

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

### 安装 Scala 版 MXNet

有两种方式安装 MXNet Scala 语言包：

* 使用预编译的二进制包

* 通过源码编译

#### 使用预编译的二进制包
对于 Linux 用户，MXNet 提供了编译好的二进制包，同时支持 CPU 和 GPU。可以通过 ```Maven``` 来下载使用这个包，根据你的需求修改下面 Maven 依赖里的 ```artifactId``` ：

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

#### 通过源码编译
在编译 MXNet Scala 语言包前，你必须已经完成了 [编译共享库](#编译共享库)。然后在 MXNet 源码根目录运行下面命令来编译 Scala 语言包：

```bash
    make scalapkg
```

这条命令会生成一个JAR文件，里面包含了封装(assembly)，核心(core)和例子(example)。同时还会在 ```native/{your-architecture}/target directory``` 下生成一个本地库，你可以通过它配合 core 使用。

在 MXNet 根目录下运行下面的命令，可以讲 MXNet Scala 语言包安装到你本地的 Maven 仓库

```bash
    make scalainstall
```

**注意 - ** 非常非常非常欢迎你们为其他操作系统和编程语言贡献简单的安装脚本，参考贡献者指南 [community page](http://mxnet.io/community/index.html)

# 下一步
* [教程](http://mxnet.io/tutorials/index.html)
* [如何使用](http://mxnet.io/how_to/index.html)
* [架构设计](http://mxnet.io/architecture/index.html)
