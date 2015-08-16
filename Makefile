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

ifndef RABIT
	RABIT = rabit
endif

# use customized config file
include $(config)
include mshadow/make/mshadow.mk
include $(DMLC_CORE)/make/dmlc.mk

# all tge possible warning tread
WARNFLAGS= -Wall
CFLAGS = -DMSHADOW_FORCE_STREAM $(WARNFLAGS)
CFLAGS += -g -O3 -I./mshadow/ -I./dmlc-core/include -fPIC -Iinclude $(MSHADOW_CFLAGS)
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

#BIN = test/test_threaded_engine test/api_registry_test
BIN = test/api_registry_test test/test_storage
OBJ = narray_op_cpu.o
# add threaded engine after it is done
OBJCXX11 = engine.o narray.o c_api.o registry.o symbol.o operator.o fully_connect_op_cpu.o static_graph.o storage.o
CUOBJ =
SLIB = lib/libmxnet.so
ALIB = lib/libmxnet.a
LIB_DEP = $(DMLC_CORE)/libdmlc.a

ifeq ($(USE_CUDA), 1)
	CUOBJ += narray_op_gpu.o fully_connected_gpu.o
endif

.PHONY: clean all test lint doc

all: $(ALIB) $(SLIB) $(BIN)

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

storage.o: src/storage/storage.cc
engine.o: src/dag_engine/simple_engine.cc
narray.o: src/narray/narray.cc
narray_op_cpu.o: src/narray/narray_op_cpu.cc src/narray/narray_op-inl.h
narray_op_gpu.o: src/narray/narray_op_gpu.cu src/narray/narray_op-inl.h
symbol.o: src/symbol/symbol.cc
static_graph.o : src/symbol/static_graph.cc
registry.o: src/registry.cc
c_api.o: src/c_api.cc
operator.o: src/operator/static_operator_wrapper.cc
fully_connected_cpu.o: src/operator/fully_connected.cc
fully_connected_gpu.o: src/operator/fully_connected.cu


lib/libmxnet.a: $(OBJ) $(OBJCXX11) $(CUOBJ)
lib/libmxnet.so: $(OBJ) $(OBJCXX11) $(CUOBJ)

test/api_registry_test: test/api_registry_test.cc lib/libmxnet.a
test/test_storage: test/test_storage.cc lib/libmxnet.a
#test/test_threaded_engine: test/test_threaded_engine.cc api/libmxnet.a

$(BIN) :
	$(CXX) $(CFLAGS) -std=c++0x -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(OBJCXX11) :
	$(CXX) -std=c++0x -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(SLIB) :
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(ALIB):
	ar cr $@ $+

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)


lint:
	python dmlc-core/scripts/lint.py mxnet ${LINT_LANG} include src scripts test python

doc:
	doxygen doc/Doxyfile

clean:
	$(RM) $(OBJ) $(OBJCXX11) $(BIN) $(CUBIN) $(CUOBJ) $(SLIB) $(ALIB) *~ */*~ */*/*~ */*/*/*~
	cd $(DMLC_CORE); make clean; cd -
