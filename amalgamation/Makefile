# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

export MXNET_ROOT=`pwd`/..

# Change this to path or specify in make command
ifndef OPENBLAS_ROOT
    export OPENBLAS_ROOT=/usr/local/opt/openblas
endif

# Whether use minimum build without blas and SSE, this will make the library super slow
ifndef MIN
	export MIN=0
	DEFS=-DMSHADOW_USE_CBLAS=1
else
	DEFS=-DMSHADOW_USE_CBLAS=0
endif

ifndef ANDROID
    export ANDROID=0
else
    DEFS+=-DMSHADOW_USE_SSE=0
endif

# Use locally installed emscripten if not specified
ifndef EMCC
    EMCC=emcc
endif

ifndef DISABLE_OPENMP
       DEFS+=-DDISABLE_OPENMP=1
endif

.PHONY: all clean

DEFS+=-DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_MKL=0 -DMSHADOW_RABIT_PS=0 -DMSHADOW_DIST_PS=0 -DDMLC_LOG_STACK_TRACE=0
DEFS+=-DMSHADOW_FORCE_STREAM -DMXNET_USE_OPENCV=0 -DMXNET_PREDICT_ONLY=1
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall $(DEFS)
ifneq ($(MIN), 1)
	CFLAGS += -I${OPENBLAS_ROOT} -I${OPENBLAS_ROOT}/include
	LDFLAGS+= -L${OPENBLAS_ROOT} -L${OPENBLAS_ROOT}/lib
        
	# Define which blas is installed. Uses OpenBLAS by default.
	ifeq ($(USE_BLAS), atlas)
                LDFLAGS += -lcblas
        else ifeq ($(USE_BLAS), blas)
                LDFLAGS += -lblas
        else
		LDFLAGS += -lopenblas
	endif
endif


all: android libmxnet_predict.a ${MXNET_ROOT}/lib/libmxnet_predict.so

nnvm.d:
	./prep_nnvm.sh

dmlc.d: dmlc-minimum0.cc
	${CXX} ${CFLAGS} -M -MT dmlc-minimum0.o \
	-I ${MXNET_ROOT}/dmlc-core/include \
	-D__MIN__=$(MIN) $+ > dmlc.d


mxnet_predict0.d: mxnet_predict0.cc nnvm.d dmlc.d
	${CXX} ${CFLAGS} -M -MT mxnet_predict0.o \
	-I ${MXNET_ROOT}/ -I ${MXNET_ROOT}/mshadow/ -I ${MXNET_ROOT}/dmlc-core/include -I ${MXNET_ROOT}/dmlc-core/src \
	-I ${MXNET_ROOT}/nnvm/include \
	-I ${MXNET_ROOT}/dlpack/include \
	-I ${MXNET_ROOT}/include \
	-D__MIN__=$(MIN) mxnet_predict0.cc > mxnet_predict0.d
	cat dmlc.d >> mxnet_predict0.d
	cat nnvm.d >> mxnet_predict0.d

mxnet_predict-all.cc:  mxnet_predict0.d dmlc-minimum0.cc nnvm.cc mxnet_predict0.cc
	@echo "Generating amalgamation to " $@
	python ./amalgamation.py $+ $@ $(MIN) $(ANDROID)

mxnet_predict-all.o: mxnet_predict-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c $+

libmxnet_predict.a: mxnet_predict-all.o
	ar rcs libmxnet_predict.a $+

jni_libmxnet_predict.o: mxnet_predict-all.cc jni/predictor.cc 
	${CXX} ${CFLAGS} -fPIC -o $@ -c jni/predictor.cc

jni_libmxnet_predict.so: jni_libmxnet_predict.o
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

ifneq ($(ANDROID), 1)
        android:
else
        CFLAGS+=  -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -O3
        LDFLAGS+=  -Wl,--no-warn-mismatch -lm_hard
        android: jni_libmxnet_predict.so
endif

libmxnet_predict.js: mxnet_predict-all.cc
	${EMCC} -std=c++11 -O2 $(DEFS) -DMSHADOW_USE_SSE=0 -D__MXNET_JS__  -o $@ $+ \
	-s EXPORTED_FUNCTIONS="['_MXPredCreate', \
	                        '_MXPredGetOutputShape', \
	                        '_MXPredSetInput', \
	                        '_MXPredForward', \
	                        '_MXPredPartialForward', \
	                        '_MXPredGetOutput', \
	                        '_MXPredFree', \
	                        '_MXNDListCreate', \
	                        '_MXNDListGet', \
	                        '_MXNDListFree']" \
	-s ALLOW_MEMORY_GROWTH=1

${MXNET_ROOT}/lib/libmxnet_predict.so:  mxnet_predict-all.o
	@mkdir -p ${MXNET_ROOT}/lib
	${CXX} ${CFLAGS} -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)
	ls -alh $@

clean:
	rm -f *.d *.o *.so *.a *.js *.js.mem mxnet_predict-all.cc nnvm.cc
