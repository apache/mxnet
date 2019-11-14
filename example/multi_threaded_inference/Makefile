CFLAGS=-std=c++11 -g -Wno-unknown-pragmas -Wall -DMXNET_USE_CUDA=1 -DMXNET_USE_CUDNN=1 -DMXNET_USE_MKLDNN=1

export MXNET_ROOT = `pwd`/../..
export CPP_PACKAGE = $(MXNET_ROOT)/cpp-package

CFLAGS += `pkg-config --cflags opencv`
LDFLAGS += `pkg-config --libs opencv`

ifndef USE_CUDA_PATH
    export USE_CUDA_PATH = /usr/local/cuda
endif

ifndef MKLDNN_BUILD_DIR
    export MKLDNN_BUILD_DIR = $(MXNET_ROOT)/3rdparty/mkldnn/build
    # Cmake build path by default
    # Uncomment below line for CMake build
    #export MKLDNN_BUILD_DIR = $(MXNET_ROOT)/build/3rdparty/mkldnn
endif

ifndef MKLDNN_INCLUDE_DIR
    export MKLDNN_INCLUDE_DIR = $(MXNET_ROOT)/3rdparty/mkldnn/include
    # Cmake build path by default
    # Uncomment below line for CMake build
    #export MKLDNN_INCLUDE_DIR = $(MXNET_ROOT)/3rdparty/mkldnn/include
endif

CFLAGS += -I$(MXNET_ROOT)/include -I$(CPP_PACKAGE)/include -I$(USE_CUDA_PATH)/include -I$(MKLDNN_INCLUDE_DIR) -I$(MKLDNN_BUILD_DIR)/include

# If MXNET_LIB_DIR env variable set use that, otherwise defaults to MXNET_ROOT/build
ifndef MXNET_LIB_DIR
    MXNET_LIB_DIR=$(MXNET_ROOT)/lib
    # cmake default by default
    # Uncomment below line for CMake build
    #MXNET_LIB_DIR=$(MXNET_ROOT)/build
endif
LDFLAGS += $(MXNET_LIB_DIR)/libmxnet.so -lpthread -L$(MKLDNN_BUILD_DIR)/src -lmkldnn -Wl,-rpath,'$${ORIGIN}'

multi_threaded_inference: multi_threaded_inference.o
	g++ -O3 -o multi_threaded_inference multi_threaded_inference.o $(LDFLAGS)

multi_threaded_inference.o: multi_threaded_inference.cc
	g++ -O3 -c multi_threaded_inference.cc $(CFLAGS)

clean:
	rm multi_threaded_inference
	rm -rf *.d *.o

lint:
	python ../../../3rdparty/dmlc-core/scripts/lint.py mxnet "cpp" ./
