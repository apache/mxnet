# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

# Added for opencv
CFLAGS+= `pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

# Added for mxnet
export MXNET_ROOT=`pwd`/../../../../mxnet

CFLAGS+=-Wall -I$(MXNET_ROOT)/include
LDFLAGS+=$(MXNET_ROOT)/lib/libmxnet.so

image-classification-predict: image-classification-predict.o
	g++ -O3 -o image-classification-predict image-classification-predict.o $(LDFLAGS)

image-classification-predict.o: image-classification-predict.cc
	g++ -O3 -c image-classification-predict.cc ${CFLAGS}
	
clean: 
	rm image-classification-predict
	rm -f *.d *.o

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./
