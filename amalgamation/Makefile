export MXNET_ROOT=`pwd`/..
# Change this to path of openblas
export OPENBLAS_ROOT=`pwd`/OpenBLAS

# Whether use minimum build without blas and SSE, this will make the library super slow
ifndef MIN
	export MIN= 0
endif

.PHONY: all clean

CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall
LDFLAGS=-lrt

ifneq ($(MIN), 1)
	CFLAGS+= -I${OPENBLAS_ROOT}
	LDFLAGS+=-L${OPENBLAS_ROOT} -lopenblas
endif

all: libmxnet_predict.a ${MXNET_ROOT}/lib/libmxnet_predict.so

mxnet_predict0.d: mxnet_predict0.cc
	${CXX} ${CFLAGS} -MD -MF $@ \
	-I ${MXNET_ROOT}/ -I ${MXNET_ROOT}/mshadow/ -I ${MXNET_ROOT}/dmlc-core/include \
	-I ${MXNET_ROOT}/include -c $+
	rm mxnet_predict0.o

mxnet_predict-all.cc:  mxnet_predict0.d mxnet_predict0.cc
	@echo "Generating amalgamation to " $@
	python ./amalgamation.py $+ $@ $(MIN)

mxnet_predict-all.o: mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c $+

libmxnet_predict.a: mxnet_predict-all.o
	ar rcs libmxnet_predict.a $+

${MXNET_ROOT}/lib/libmxnet_predict.so:  mxnet_predict-all.o
	@mkdir -p ${MXNET_ROOT}/lib
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)
	ls -alh $@

clean:
	rm -f *.d *.o
