USE_CUDA=-lcudnn -lnvrtc -lcuda -lcudart -lcublas -lcurand -DMSHADOW_USE_CUDA=1 -DMSHADOW_USE_CUDNN=1 
BLAS=-lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 
OPTIONS=-DMSHADOW_RABIT_PS=0 -DMSHADOW_DIST_PS=0 -DMXNET_USE_OPENCV=1

CXX=g++

CFLAGS=-I ../../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp -I ../../mshadow -I ../../dmlc-core/include
LDFLAGS=-L ../../lib -lmxnet -lpthread -lm $(BLAS) -lrt `pkg-config --cflags opencv` $(USE_CUDA) $(OPTIONS)

COMPILE_com=$(CXX) -std=c++0x $(CFLAGS) $(LDFLAGS)

#####
mlp: ./mlp.cpp
	$(COMPILE_com)  -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^)
mlp_run: mlp
	LD_LIBRARY_PATH=../../lib ./mlp

#####
use_ndarray: ./use_ndarray.cpp
	$(COMPILE_com)  -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^)
use_ndarray_run: use_ndarray
	LD_LIBRARY_PATH=../../lib ./use_ndarray
