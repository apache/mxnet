export MXNET_ROOT=`pwd`/..
# Change this to path of openblas
export OPENBLAS_ROOT=${MXNET_ROOT}/../OpenBLAS
# Change this to NNPACK path if you need
USE_NNPACK_NUM_THREADS=4
export NNPACK_ROOT=${MXNET_ROOT}/../NNPACK

# Whether use minimum build without blas and SSE, this will make the library super slow
ifndef MIN
	export MIN=0
endif

ifndef ANDROID
    export ANDROID=0
endif

ifndef NNPACK
	export NNPACK=0
endif

ifndef MKL2017
	export MKL2017=0
endif


.PHONY: all clean


CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall
ifneq ($(MIN), 1)
	CFLAGS += -I${OPENBLAS_ROOT} -I${OPENBLAS_ROOT}/include
	LDFLAGS+= -L${OPENBLAS_ROOT} -L${OPENBLAS_ROOT}/lib -lopenblas
endif

ifneq ($(NNPACK), 0)
	CFLAGS += -I${NNPACK_ROOT}/include
	CFLAGS += -I${NNPACK_ROOT}/third-party/FXdiv/include
	CFLAGS += -I${NNPACK_ROOT}/third-party/pthreadpool/include
	CFLAGS += -DMXNET_USE_NNPACK=1
	CFLAGS += -DMXNET_USE_NNPACK_NUM_THREADS=$(USE_NNPACK_NUM_THREADS)
	LDFLAGS += -L${NNPACK_ROOT}/lib -lnnpack
endif

ifneq ($(MKL2017), 0)
	CFLAGS += -DMXNET_USE_MKL2017=1
	CFLAGS += -DUSE_MKL=1
	CFLAGS += -I$(MXNET_ROOT)/src/operator/mkl/
	CFLAGS += -DMKL_EXPERIMENTAL=0
endif

all: android libmxnet_predict.a ${MXNET_ROOT}/lib/libmxnet_predict.so

mxnet_predict0.d: mxnet_predict0.cc
	${CXX} ${CFLAGS} -MD -MF $@ \
	-I ${MXNET_ROOT}/ -I ${MXNET_ROOT}/mshadow/ -I ${MXNET_ROOT}/dmlc-core/include -I ${MXNET_ROOT}/nnvm/include \
	-I ${MXNET_ROOT}/include \
	-D__MIN__=$(MIN) -c $+
	rm mxnet_predict0.o

mxnet_predict-all.cc:  mxnet_predict0.d mxnet_predict0.cc
	@echo "Generating amalgamation to " $@
	python ./amalgamation.py $+ $@ $(MIN) $(ANDROID) $(NNPACK) $(MKL2017)

mxnet_predict-all.o: mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c $+

libmxnet_predict.a: mxnet_predict-all.o
	ar rcs libmxnet_predict.a $+

jni_libmxnet_predict.o: mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c jni/predictor.cc

jni_libmxnet_predict.so: jni_libmxnet_predict.o
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)


ifneq ($(ANDROID), 1)
        LDFLAGS+= -lrt
        android:
else
        CFLAGS+=  -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -O3
        LDFLAGS+=  -Wl,--no-warn-mismatch -lm_hard
        android: jni_libmxnet_predict.so
endif

libmxnet_predict.js: mxnet_predict-all.cc
	emcc -std=c++11 -O2 -D__MXNET_JS__  -o $@ $+ \
	-s EXPORTED_FUNCTIONS="['_MXPredCreate', '_MXPredGetOutputShape', '_MXPredSetInput', '_MXPredForward', '_MXPredPartialForward', '_MXPredGetOutput', '_MXPredFree', '_MXNDListCreate', '_MXNDListGet', '_MXNDListFree']" \
	-s ALLOW_MEMORY_GROWTH=1


${MXNET_ROOT}/lib/libmxnet_predict.so:  mxnet_predict-all.o
	@mkdir -p ${MXNET_ROOT}/lib
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)
	ls -alh $@

clean:
	rm -f *.d *.o *.so *.a mxnet_predict-all.cc
