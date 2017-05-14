A Foolproof Guide to MXNet R Installation
===================

We introduce how to install MXNet R on a Linux step by step.

We assume Linux and R are brand new without any extra packages and  tools installed. Assume MXNet is working.

The code is  tested on CentOS 7 and R 3.2.3.

1. Preparation
--

1. Meet dependency requirements by `devtool`
```
#1
#CentOS
sudo yum install  openssl-devel -y

#2
#Ubuntu 
sudo apt-get -y build-dep libcurl4-gnutls-dev
sudo apt-get -y install libcurl4-gnutls-dev

#CentOS
sudo yum -y install libcurl libcurl-devel

#3
#R console
install.packages("httr")
install.packages("git2r")
install.packages("curl")
```

2. Installation
--
```
cd ~/Downloads/mxnet #Change to your mxnet root folder
Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
cd ..
make rpkg
R CMD INSTALL mxnet_0.5.tar.gz
```

3. Test & Troubleshooting
--
```
# Test
require(mxnet)

Loading required package: mxnet
Error : .onLoad failed in loadNamespace() for 'mxnet', details:
  call: dyn.load(file, DLLpath = DLLpath, ...)
  error: unable to load shared object '.../R/x86_64-redhat-linux-gnu-library/3.2/mxnet/libs/libmxnet.so':
  libcudart.so.7.5: cannot open shared object file: No such file or directory
```

Solution:
1. Make sure the environment is set up in `.bashrc` 
```
# Terminal
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib
# Note the CUDA version varies
```
2.  If not work, try
```
sudo ldconfig /usr/local/cuda/lib64
```
