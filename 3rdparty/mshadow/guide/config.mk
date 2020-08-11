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

#---------------------------------------------------------------------------------------
#  mshadow: the configuration compile script
#
#  This is configuration script that you can use to compile mshadow
#  Usage:
#
#  include config.mk in your Makefile, or directly include the definition of variables
#  include mshadow.mk after the variables are set
#
#  Add MSHADOW_CFLAGS to the compile flags
#  Add MSHADOW_LDFLAGS to the linker flags
#  Add MSHADOW_NVCCFLAGS to the nvcc compile flags
#----------------------------------------------------------------------------------------

# whether use CUDA during compile
USE_CUDA = 0

# add the path to CUDA libary to link and compile flag
# if you have already add them to enviroment variable, leave it as NONE
USE_CUDA_PATH = NONE

#
# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas, apple
USE_BLAS = blas
#
# add path to intel library, you may need it
# for MKL, if you did not add the path to enviroment variable
#
USE_INTEL_PATH = NONE

# whether compile with parameter server
USE_DIST_PS = 0
PS_PATH = NONE
PS_THIRD_PATH = NONE

# whether compile with rabit allreduce
USE_RABIT_PS = 0
RABIT_PATH = NONE
