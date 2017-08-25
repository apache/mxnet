CXX=g++
BLAS=-L /opt/openblas/lib -lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 
CUDA=-DMSHADOW_USE_CUDA=1
OPENCV_CFLAGS=`pkg-config --cflags opencv`
OPENCV_LDFLAGS=`pkg-config --libs opencv`

#COMMFLAGS=-static -static-libgcc -static-libstdc++

CFLAGS=$(COMMFLAGS) -I ../../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp 
LDFLAGS=$(COMMFLAGS) -L ../../lib/linux -lmxnet $(BLAS) $(CUDA) -lgomp -pthread

all: feature_extract prepare_data_with_opencv

feature_extract: ./feature_extract.cpp
	$(CXX) -c -std=c++0x $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

prepare_data_with_opencv: ./prepare_data_with_opencv.cpp
	$(CXX) -c -std=c++0x $(OPENCV_CFLAGS) $^ 
	$(CXX) $(basename $@).o -o $@ $(OPENCV_LDFLAGS)
	-rm -f $(basename $@).o

clean:
	-rm -f feature_extract
	-rm -f prepare_data_with_opencv
