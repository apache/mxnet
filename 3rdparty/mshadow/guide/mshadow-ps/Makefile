# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
include config.mk
include ../../make/mshadow.mk
export CFLAGS = -Wall -O3 -std=c++11 -fopenmp -I../../ $(MSHADOW_CFLAGS)
export LDFLAGS= -lm $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)

# specify tensor path local_sum.cpu
BIN = local_sum.cpu
OBJ =
CUOBJ =
CUBIN = local_sum.gpu

ifeq ($(USE_DIST_PS),1)
BIN = dist_async_sum.cpu
LDFLAGS += -lunwind
endif

.PHONY: clean all

all: $(BIN) #$(CUBIN)

local_sum.cpu: local_sum.cpp
local_sum.gpu: local_sum.cu

dist_async_sum.cpu: dist_async_sum.cpp dist_async_sum-inl.h
dist_sync_sum.cpu: dist_sync_sum.cpp

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)  $(LDFLAGS) $(PS_LIB)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~
