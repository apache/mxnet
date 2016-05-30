# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

# Added for opencv
# CFLAGS+= `pkg-config --cflags opencv`
# LDFLAGS+=`pkg-config --libs opencv`

# Added for mxnet
export MXNET_ROOT=/home/lisheng.ls/mxnet

CFLAGS+= -I$(MXNET_ROOT)/include 
LDFLAGS+=$(MXNET_ROOT)/lib/libmxnet.so

lstm-word-segment-predict: lstm-word-segment-predict.o
	g++ -O3 -o lstm-word-segment-predict lstm-word-segment-predict.o $(LDFLAGS)

lstm-word-segment-predict.o: lstm-word-segment-predict.cc
	g++ -O3 -c lstm-word-segment-predict.cc ${CFLAGS}
	
clean: 
	rm lstm-word-segment-predict
	rm -f *.d *.o

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./
