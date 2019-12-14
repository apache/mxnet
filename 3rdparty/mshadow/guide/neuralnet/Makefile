# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
include config.mk
include ../../make/mshadow.mk
export CFLAGS = -Wall -O3 -I../../ -fopenmp $(MSHADOW_CFLAGS)
export LDFLAGS= -lm $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)

# specify tensor path
BIN =
OBJ =
CUOBJ =
CUBIN = nnet convnet nnet_ps
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

nnet: nnet.cu
nnet_ps: nnet_ps.cu
convnet: convnet.cu

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)  $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~

