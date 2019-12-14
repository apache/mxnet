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
USE_CUDA = 1

# add the path to CUDA libary to link and compile flag
# if you have already add them to enviroment variable, leave it as NONE
USE_CUDA_PATH = NONE

#
# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas, apple
USE_BLAS = openblas
#
# add path to intel library, you may need it
# for MKL, if you did not add the path to enviroment variable
#
USE_INTEL_PATH = NONE

# whether compile with parameter server
USE_DIST_PS = 0
PS_PATH = NONE
PS_THIRD_PATH = NONE
