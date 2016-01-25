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

# whether compile with debug
DEBUG = 0

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

# whether use CUDNN R3 library
USE_CUDNN = 0

# whether use cuda runtime compiling for writing kernels in native language (i.e. Python)
USE_NVRTC = 0

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 1

# use openmp for parallelization
USE_OPENMP = 0

# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
USE_BLAS = apple

# add path to intel library, you may need it for MKL, if you did not add the path
# to environment variable
USE_INTEL_PATH = NONE

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
# plugins
#----------------------------

# whether to use torch integration. This requires installing torch.
# TORCH_PATH = $(HOME)/torch
# MXNET_PLUGINS += plugin/torch/torch.mk
