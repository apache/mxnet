ifndef config
ifdef CXXNET_CONFIG
	config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif

ifndef DMLC_CORE
	DMLC_CORE = dmlc-core
endif


# use customized config file
include $(config)
include mshadow/make/mshadow.mk
include $(DMLC_CORE)/make/dmlc.mk

# all tge possible warning tread
WARNFLAGS= -Wall
CFLAGS = -DMSHADOW_FORCE_STREAM $(WARNFLAGS)

# CFLAGS for debug
ifeq ($(DEBUG),0)
	CFLAGS += -O3
else
	CFLAGS += -g -O0
endif
CFLAGS += -I./mshadow/ -I./dmlc-core/include -fPIC -Iinclude $(MSHADOW_CFLAGS)
LDFLAGS = -pthread $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)
NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
ROOTDIR = $(CURDIR)

ifndef LINT_LANG
	LINT_LANG="all"
endif

# setup opencv
ifeq ($(USE_OPENCV),1)
	CFLAGS+= -DCXXNET_USE_OPENCV=1
	LDFLAGS+= `pkg-config --libs opencv`
else
	CFLAGS+= -DCXXNET_USE_OPENCV=0
endif

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DCXXNET_USE_CUDNN=1
	LDFLAGS += -lcudnn
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

#ENGINE=simple_engine.o dag_engine.o
ENGINE=naive_engine.o
BIN = tests/test_simple_engine
OBJ = narray_function_cpu.o
OBJCXX11 = narray.o c_api.o operator.o symbol.o storage.o static_graph.o graph_executor.o io.o iter_mnist.o $(ENGINE)
CUOBJ = narray_function_gpu.o
SLIB = lib/libmxnet.so
ALIB = lib/libmxnet.a
LIB_DEP = $(DMLC_CORE)/libdmlc.a
ALL_DEP = $(OBJ) $(OBJCXX11) $(LIB_DEP)
# common headers, change them will results in rebuild of all files
COMMON_HEADERS=include/mxnet/*.h src/common/*.h

.PHONY: clean all test lint doc

all: $(ALIB) $(SLIB) $(BIN)

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

storage.o: src/storage/storage.cc
naive_engine.o:  src/dag_engine/naive_engine.cc
dag_engine.o: src/dag_engine/dag_engine.cc
simple_engine.o: src/dag_engine/simple_engine.cc
narray.o: src/narray/narray.cc
narray_function_cpu.o: src/narray/narray_function.cc src/narray/narray_function-inl.h
narray_function_gpu.o: src/narray/narray_function.cu src/narray/narray_function-inl.h
symbol.o: src/symbol/symbol.cc src/symbol/*.h
graph_executor.o: src/symbol/graph_executor.cc src/symbol/*.h
static_graph.o : src/symbol/static_graph.cc src/symbol/*.h
operator.o: src/operator/operator.cc
c_api.o: src/c_api.cc
io.o: src/io/io.cc
iter_mnist.o: src/io/iter_mnist.cc src/io/*.h

# Rules for operators
OPERATOR_HDR=$(wildcard src/operator/*-inl.h)
OPERATOR_OBJ=$(patsubst %-inl.h, %_cpu.o, $(OPERATOR_HDR))
OPERATOR_CUOBJ=$(patsubst %-inl.h, %_gpu.o, $(OPERATOR_HDR))

ALL_DEP += $(OPERATOR_OBJ)
ifeq ($(USE_CUDA), 1)
	ALL_DEP += $(OPERATOR_CUOBJ) $(CUOBJ)
endif

src/operator/%_cpu.o : src/operator/%.cc src/operator/%-inl.h src/operator/mshadow_op.h src/operator/operator_common.h $(COMMON_HEADERS)
	$(CXX) -std=c++0x -c $(CFLAGS) -o $@  $(filter %.cpp %.c %.cc, $^)

src/operator/%_gpu.o : src/operator/%.cu src/operator/%-inl.h src/operator/operator_common.h src/operator/mshadow_op.h $(COMMON_HEADERS)
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

lib/libmxnet.a:  $(ALL_DEP)
lib/libmxnet.so: $(ALL_DEP)

tests/test_storage: tests/test_storage.cc lib/libmxnet.a
tests/test_simple_engine: tests/test_simple_engine.cc lib/libmxnet.a

$(BIN) :
	$(CXX) $(CFLAGS) -std=c++0x -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(OBJ) : $(COMMON_HEADERS)
	$(CXX) -c $(CFLAGS) -o $@ $(filter %.cpp %.c %.cc, $^)

$(OBJCXX11) : $(COMMON_HEADERS)
	$(CXX) -std=c++0x -c $(CFLAGS) -o $@  $(filter %.cpp %.c %.cc, $^)

$(SLIB) :
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(ALIB): $(OBJ) $(OBJCXX11)
	ar cr $@ $+

$(CUOBJ) :$(COMMON_HEADERS)
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

lint:
	python dmlc-core/scripts/lint.py mxnet ${LINT_LANG} include src scripts python

doxygen:
	doxygen doc/Doxyfile

clean:
	$(RM) $(ALL_DEP) $(SLIB) $(ALIB) *~ */*~ */*/*~ */*/*/*~
	cd $(DMLC_CORE); make clean; cd -
