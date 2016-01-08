CFLAGS=-I ../../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp -I ../../mshadow -I ../../dmlc-core/include
LDFLAGS=-L ../../lib -lmxnet -lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CUDA=1

CXX=g++

mlp: ./mlp.cpp
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $^ $(LDFLAGS)

use_ndarray: ./use_ndarray.cpp
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $^ $(LDFLAGS)

lint:
	python2 ../../dmlc-core/scripts/lint.py mxnet "cpp" ./

clean:
	rm -f mlp use_ndarray
