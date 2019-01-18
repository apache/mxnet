# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------------------------
#  Template configuration for compiling mxnet for making python wheel
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
USE_PROFILER = 1

# whether to turn on signal handler (e.g. segfault logger)
USE_SIGNAL_HANDLER = 1

# the additional link flags you want to add
ADD_LDFLAGS += -L$(DEPS_PATH)/lib $(DEPS_PATH)/lib/libculibos.a -lpng -ltiff -ljpeg -lz -ldl -lgfortran -Wl,--version-script=$(CURDIR)/make/config/libmxnet.ver,-rpath,'$${ORIGIN}',--gc-sections

# the additional compile flags you want to add
ADD_CFLAGS += -I$(DEPS_PATH)/include -ffunction-sections -fdata-sections

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------

# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
# in default use atlas for linux while apple for osx
USE_BLAS=openblas

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 1

# whether use CUDA during compile
USE_CUDA = 1

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable, leave it as NONE
# USE_CUDA_PATH = /usr/local/cuda
USE_CUDA_PATH = $(DEPS_PATH)/usr/local/cuda-8.0

# whether to use CuDNN library
USE_CUDNN = 1

# whether to use NCCL library
USE_NCCL = 1

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
# CUDA_ARCH :=

# whether use cuda runtime compiling for writing kernels in native language (i.e. Python)
USE_NVRTC = 1

# use openmp for parallelization
USE_OPENMP = 1
USE_OPERATOR_TUNING = 1
USE_LIBJPEG_TURBO = 1

# whether use MKL-DNN library
USE_MKLDNN = 1

# whether use NNPACK library
USE_NNPACK = 0

# whether use lapack during compilation
# only effective when compiled with blas versions openblas/apple/atlas/mkl
USE_LAPACK = 1

# path to lapack library in case of a non-standard installation
USE_LAPACK_PATH = $(DEPS_PATH)/lib

# add path to intel library, you may need it for MKL, if you did not add the path
# to environment variable
USE_INTEL_PATH = NONE

# If use MKL, choose static link automatically to allow python wrapper
ifeq ($(USE_BLAS), mkl)
USE_STATIC_MKL = 1
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
USE_DIST_KVSTORE = 1

# whether or not allow to read and write HDFS directly. If yes, then hadoop is
# required
USE_HDFS = 0

# path to libjvm.so. required if USE_HDFS=1
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server

# whether or not allow to read and write AWS S3 directly. If yes, then
# libcurl4-openssl-dev is required, it can be installed on Ubuntu by
# sudo apt-get install -y libcurl4-openssl-dev
USE_S3 = 1

#----------------------------
# additional operators
#----------------------------

# path to folders containing projects specific operators that you don't want to put in src/operators
EXTRA_OPERATORS =


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
