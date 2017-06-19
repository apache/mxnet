#-------------------------------------------------------------------------------
#  Template configuration for compiling mxnet
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of mxnet. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#-------------------------------------------------------------------------------

#---------------------
# choice of compiler
#--------------------

export CC = gcc
export CXX = g++
export NVCC = nvcc

# whether compile with options for MXNet developer
DEV = 0

# whether compile with debug
DEBUG = 0

# whether compiler with profiler
USE_PROFILER =

# the additional link flags you want to add
ADD_LDFLAGS =

# the additional compile flags you want to add
ADD_CFLAGS =

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------

# whether use CUDA during compile
USE_CUDA = 0

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable, leave it as NONE
# USE_CUDA_PATH = /usr/local/cuda
USE_CUDA_PATH = NONE

# whether use CuDNN R3 library
USE_CUDNN = 0

# whether use cuda runtime compiling for writing kernels in native language (i.e. Python)
USE_NVRTC = 0

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 1

# use openmp for parallelization
USE_OPENMP = 1

# MKL ML Library for Intel CPU/Xeon Phi
# Please refer to MKL_README.md for details

# MKL ML Library folder, need to be root for /usr/local
# Change to User Home directory for standard user
# For USE_BLAS!=mkl only
MKLML_ROOT=/usr/local

# whether use MKL2017 library
USE_MKL2017 = 0

# whether use MKL2017 experimental feature for high performance
# Prerequisite USE_MKL2017=1
USE_MKL2017_EXPERIMENTAL = 0

# whether use NNPACK library
USE_NNPACK = 0

# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
# in default use atlas for linux while apple for osx
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
USE_BLAS = apple
else
USE_BLAS = atlas
endif

# whether use lapack during compilation
# only effective when compiled with blas versions openblas/apple/atlas/mkl
USE_LAPACK = 1

# path to lapack library in case of a non-standard installation
USE_LAPACK_PATH =

# add path to intel library, you may need it for MKL, if you did not add the path
# to environment variable
USE_INTEL_PATH = NONE

# If use MKL only for BLAS, choose static link automatically to allow python wrapper
ifeq ($(USE_MKL2017), 0)
ifeq ($(USE_BLAS), mkl)
USE_STATIC_MKL = 1
endif
else
USE_STATIC_MKL = NONE
endif

#----------------------------
# Settings for power and arm arch
#----------------------------
ARCH := $(shell uname -a)
ifneq (,$(filter $(ARCH), armv6l armv7l powerpc64le ppc64le aarch64))
	USE_SSE=0
else
	USE_SSE=1
endif

#----------------------------
# distributed computing
#----------------------------

# whether or not to enable multi-machine supporting
USE_DIST_KVSTORE = 0

# whether or not allow to read and write HDFS directly. If yes, then hadoop is
# required
USE_HDFS = 0

# path to libjvm.so. required if USE_HDFS=1
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server

# whether or not allow to read and write AWS S3 directly. If yes, then
# libcurl4-openssl-dev is required, it can be installed on Ubuntu by
# sudo apt-get install -y libcurl4-openssl-dev
USE_S3 = 0

#----------------------------
# additional operators
#----------------------------

# path to folders containing projects specific operators that you don't want to put in src/operators
EXTRA_OPERATORS =

#----------------------------
# other features
#----------------------------

# Create C++ interface package
USE_CPP_PACKAGE = 0

#----------------------------
# plugins
#----------------------------

# whether to use caffe integration. This requires installing caffe.
# You also need to add CAFFE_PATH/build/lib to your LD_LIBRARY_PATH
# CAFFE_PATH = $(HOME)/caffe
# MXNET_PLUGINS += plugin/caffe/caffe.mk

# whether to use torch integration. This requires installing torch.
# You also need to add TORCH_PATH/install/lib to your LD_LIBRARY_PATH
# TORCH_PATH = $(HOME)/torch
# MXNET_PLUGINS += plugin/torch/torch.mk

# WARPCTC_PATH = $(HOME)/warp-ctc
# MXNET_PLUGINS += plugin/warpctc/warpctc.mk

# whether to use sframe integration. This requires build sframe
# git@github.com:dato-code/SFrame.git
# SFRAME_PATH = $(HOME)/SFrame
# MXNET_PLUGINS += plugin/sframe/plugin.mk
