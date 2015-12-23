MXNET_ROOT=../..

### include user config
ifndef config
ifdef CXXNET_CONFIG
	config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard $(MXNET_ROOT)/config.mk)","")
	config = $(MXNET_ROOT)/config.mk
else
	config = $(MXNET_ROOT)/make/config.mk
endif
endif
include $(config)

### blas library
ifeq ($(USE_BLAS), mkl)
ifneq ($(USE_INTEL_PATH), NONE)
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		MSHADOW_LDFLAGS += -L$(USE_INTEL_PATH)/mkl/lib
		MSHADOW_LDFLAGS += -L$(USE_INTEL_PATH)/lib
	else
		MSHADOW_LDFLAGS += -L$(USE_INTEL_PATH)/mkl/lib/intel64
		MSHADOW_LDFLAGS += -L$(USE_INTEL_PATH)/lib/intel64
	endif
	MSHADOW_CFLAGS += -I$(USE_INTEL_PATH)/mkl/include
endif
ifneq ($(USE_STATIC_MKL), NONE)
ifeq ($(USE_INTEL_PATH), NONE)
	MKLROOT = /opt/intel/mkl
else
	MKLROOT = $(USE_INTEL_PATH)/mkl
endif
	MSHADOW_LDFLAGS +=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -ldl -lpthread -lm
else
	MSHADOW_LDFLAGS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
endif
else
ifneq ($(USE_BLAS), NONE)
	MSHADOW_CFLAGS += -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
endif
endif
ifeq ($(USE_BLAS), openblas)
	MSHADOW_LDFLAGS += -lopenblas
else ifeq ($(USE_BLAS), atlas)
	MSHADOW_LDFLAGS += -lcblas
else ifeq ($(USE_BLAS), blas)
	MSHADOW_LDFLAGS += -lblas
else ifeq ($(USE_BLAS), apple)
	MSHADOW_CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/
	MSHADOW_LDFLAGS += -framework Accelerate
endif

### compile flags
CFLAGS=-I $(MXNET_ROOT)/include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -I $(MXNET_ROOT)/mshadow -I $(MXNET_ROOT)/dmlc-core/include $(MSHADOW_CFLAGS)
LDFLAGS=-L $(MXNET_ROOT)/lib -lmxnet $(MSHADOW_LDFLAGS)

ifeq ($(USE_CUDA), 1)
	LDFLAGS += -lnvrtc -lcuda
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
endif

all: image_predict mlp use_ndarray mnist

image_predict:
	cd ./image-classification;make;cp ./image-classification-predict ../image_predict;cd -

mlp: ./symbol-ndarray/mlp.cc
	g++ -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@ $^

use_ndarray: ./symbol-ndarray/use_ndarray.cc
	g++ -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@ $^

mnist: ./mnist-src/mnist.cc
	g++ -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@ $^

lint:
	python2 ../../dmlc-core/scripts/lint.py mxnet "cpp" ./

clean:
	rm image_predict mlp use_ndarray mnist
