# 在 OS X (Mac) 上安装 MXNet
MXNet 目前支持 Python, R, Julia, and Scala。对于 Mac 上使用 Python 的用户，MXNet提供了一个安装全部 MXNet 依赖和 MXNet 库的 Git Bash 脚本。

## 为GPU版本安装准备环境

这部分是可选的。如果不打算使用GPU的话可以跳过此部分。如果使用GPU你需要配置 CUDA 和 cuDNN。

首先,下载并安装 [CUDA 8 工具包](https://developer.nvidia.com/cuda-toolkit).

安装完了CUDA工具包你需要配置一些环境变量，将下面命令添加到 ```~/.bash_profile``` 文件中：

```bash
    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH="$CUDA_HOME/lib:$DYLD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
```

重新加载 ```~/.bash_profile``` 文件然后安装依赖：

```bash
    . ~/.bash_profile
    brew install coreutils
    brew tap caskroom/cask
```

然后下载 [cuDNN 5](https://developer.nvidia.com/cudnn).

解压此文件并且进入 cudnn 根目录。将头文件和库文件移动到你本地的 CUDA 工具包文件夹:

```bash
    $ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-8.0/include/
    $ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-8.0/lib
    $ sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn* /usr/local/cuda/lib/
```

现在我们可以开始编译 MXNet.

## 快速安装
### 安装 Python 版 MXNet

克隆MXNet的源码仓库到你的电脑上，运行安装脚本。除了安装 MXNet 外，这个脚本还会安装： ```Homebrew```, ```Numpy```, ```LibBLAS```, ```OpenCV```, ```Graphviz```, ```NumPy``` 以及 ```Jupyter``` 。

安装这些大概需要花费5到10分钟。

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
    bash install-mxnet-osx-python.sh
```

点击 [这里](https://raw.githubusercontent.com/dmlc/mxnet/master/setup-utils/install-mxnet-osx-python.sh) 可以查看安装脚本的具体内容。

## 标准安装

安装 MXNet 一共分两步：

1. 将 MXNet C++ 代码编译成共享库。
2. 为 MXNet 安装特定语言包。

**注意:** 可以通过编辑 ```make/config.mk``` 来修改编译选项，然后通过 ```make``` 命令开始编译。

### 编译共享库

#### 安装 MXNet 依赖：

用下面命令安装依赖库：

- [Homebrew](http://brew.sh/)
- OpenBLAS and homebrew/science (线性代数计算)
- OpenCV (机器视觉处理)

```bash
	# Paste this command in Mac terminal to install Homebrew
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

	# Insert the Homebrew directory at the top of your PATH environment variable
	export PATH=/usr/local/bin:/usr/local/sbin:$PATH
```

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

#### 编译 MXNet 共享库

安装完依赖后，从Git上下载 MXNet 源码，然后编译成库文件名字叫 ```libmxnet.so```。

 OS X 上编译 MXNet 的配置文件叫 ```osx.mk``` 。首先将 ```make/osx.mk``` 拷贝为 ```config.mk```，这样 ```make``` 命令可以使用它：

```bash
    git clone --recursive https://github.com/dmlc/mxnet ~/mxnet
    cd ~/mxnet
    cp make/osx.mk ./config.mk
    echo "USE_BLAS = openblas" >> ./config.mk
    echo "ADD_CFLAGS += -I/usr/local/opt/openblas/include" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/" >> ./config.mk
    make -j$(sysctl -n hw.ncpu)
```

如果我们编译 ```GPU``` 版本，需要将下面的配置添加到 config.mk 中，然后再编译：

```bash
    echo "USE_CUDA = 1" >> ./config.mk
    echo "USE_CUDA_PATH = /usr/local/cuda" >> ./config.mk
    echo "USE_CUDNN = 1" >> ./config.mk
    make
```
**注意:** 想修改编译参数，编辑 ```config.mk```.


&nbsp;

我们已经安装完了 MXNet 的核心库(core library)，接下来我们要选择一种编程语言，安装语言接口包:
- [R](#安装-r-版-mxnet)
- [Julia](#安装-julia-版-mxnet)
- [Scala](#安装-scala-版-mxnet)


### 安装 R 版 MXNet
有两个选择:
1. 使用预编译的二进制包
2. 使用源码编译

#### 使用预编译的二进制包

对于 OS X (Mac) 用户, MXNet 提供了编译好的 CPU 版二进制包. 这个包每周更新，你可以在 R 控制台中使用下面的命令来安装这个包：

```r
	install.packages("drat", repos="https://cran.rstudio.com")
	drat:::addRepo("dmlc")
	install.packages("mxnet")
```

#### 使用源码编译

运行下面命令来安装 MXNet 依赖并且编译 MXNet R 语言包

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
MXNet Julia 语言包托管在一个单独的仓库 ```MXNet.jl```，地址是 [GitHub](https://github.com/dmlc/MXNet.jl)。 Julia 需要与已经安装的 libmxnet 绑定。使用下面的命令来配置 ```MXNET_HOME``` 环境变量：

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
在编译 MXNet Scala 语言包前，你必须已经完成了 [编译共享库](#编译共享库)。然后在 MXNet 源码根目录运行下面命令来编译 Scala 语言包：

```bash
    make scalapkg
```

这条命令会生成一个 JAR 文件，里面包含了封装(assembly)，核心(core)和例子(example)。同时还会在 ```native/{your-architecture}/target directory``` 下生成一个本地库，你可以通过它配合 core 使用。

在 MXNet 根目录下运行下面的命令，可以讲 MXNet Scala 语言包安装到你本地的 Maven 仓库

```bash
    make scalainstall
```

# 下一步
* [教程](http://mxnet.io/tutorials/index.html)
* [如何使用](http://mxnet.io/how_to/index.html)
* [架构设计](http://mxnet.io/architecture/index.html)
