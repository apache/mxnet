
ifndef config
ifdef CXXNET_CONFIG
  config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ../../config.mk)","")
  config = ../../config.mk
else
  config = ../../make/config.mk
endif
endif

include $(config)

BLAS=-L /opt/openblas/lib -lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 
CUDA=-DMSHADOW_USE_CUDA=1

#COMMFLAGS=-static -static-libgcc -static-libstdc++

CFLAGS=$(COMMFLAGS) -I ../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -I ../../nnvm/include -I ../../include -I ../../dmlc-core/include
LDFLAGS=$(COMMFLAGS) -L ../../lib -lmxnet $(BLAS) $(CUDA) -pthread

ifdef CAFFE_PATH
  CFLAGS  += -DMXNET_USE_CAFFE=1 -I$(CAFFE_PATH)/include -I$(CAFFE_PATH)/build/include
  LDFLAGS += -L$(CAFFE_PATH)/build/lib -lcaffe
endif

ifneq ($(OS), Windows_NT)
	OS := $(shell uname)
endif
ifneq ($(OS), Darwin)
	CFLAGS += -fopenmp
	LDFLAGS += -lgomp
endif

.PHONY: all travis clean

all: mlp lenet lenet_with_mxdataiter alexnet googlenet resnet

lenet_with_mxdataiter: ./lenet_with_mxdataiter.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

lenet: ./lenet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

mlp: ./mlp.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

alexnet: ./alexnet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

googlenet: ./googlenet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

inception_bn: ./inception_bn.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

resnet: ./resnet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

# For simplicity, no link here
travis:
	$(CXX) -c -std=c++11 $(CFLAGS) ./mlp.cpp && rm -f mlp.o
	$(CXX) -c -std=c++11 $(CFLAGS) ./lenet.cpp && rm -f lenet.o
	$(CXX) -c -std=c++11 $(CFLAGS) ./lenet_with_mxdataiter.cpp && rm -f lenet_with_mxdataiter.o
	$(CXX) -c -std=c++11 $(CFLAGS) ./alexnet.cpp && rm -f alexnet.o
	$(CXX) -c -std=c++11 $(CFLAGS) ./googlenet.cpp && rm -f googlenet.o
	$(CXX) -c -std=c++11 $(CFLAGS) ./resnet.cpp && rm -f resnet.o


clean:
	-rm -f mlp
	-rm -f lenet
	-rm -f lenet_with_mxdataiter
	-rm -f alexnet
	-rm -f googlenet
	-rm -f resnet
