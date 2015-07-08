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
CFLAGS += -g -O3 -I./mshadow/ -I./dmlc-core/include -fPIC -Iinclude $(MSHADOW_CFLAGS)
LDFLAGS = -pthread $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)
NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
ROOTDIR = $(CURDIR)

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

BIN = test/test_threaded_engine #test/api_registry_test
OBJ = storage.o narray_op_cpu.o operator.o operator_cpu.o 
OBJCXX11 = engine.o narray.o api_registry.o
CUOBJ = narray_op_gpu.o operator_gpu.o
SLIB = api/libmxnet.so
ALIB = api/libmxnet.a
LIB_DEP = $(DMLC_CORE)/libdmlc.a

.PHONY: clean all

all: $(ALIB) $(SLIB) $(BIN)

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

storage.o: src/storage/storage.cc
#engine.o: src/dag_engine/simple_engine.cc
engine.o: src/dag_engine/threaded_engine.cc src/common/concurrent_blocking_queue.h src/common/spin_lock.h
narray.o: src/narray/narray.cc
narray_op_cpu.o: src/narray/narray_op_cpu.cc src/narray/narray_op-inl.h
narray_op_gpu.o: src/narray/narray_op_gpu.cu src/narray/narray_op-inl.h
operator.o: src/operator/operator.cc
operator_cpu.o: src/operator/operator_cpu.cc
operator_gpu.o: src/operator/operator_gpu.cu
api_registry.o: src/api_registry.cc
mxnet_api.o: api/mxnet_api.cc

api/libmxnet.a: $(OBJ) $(OBJCXX11) $(CUOBJ)
api/libmxnet.so: $(OBJ) $(OBJCXX11) $(CUOBJ)

test/api_registry_test: test/api_registry_test.cc api/libmxnet.a
test/test_threaded_engine: test/test_threaded_engine.cc api/libmxnet.a

$(BIN) :
	$(CXX) $(CFLAGS) -std=c++11 -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(OBJCXX11) :
	$(CXX) -std=c++11 -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(SLIB) :
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(ALIB):
	ar cr $@ $+

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(OBJCXX11) $(BIN) $(CUBIN) $(CUOBJ) $(SLIB) $(ALIB) *~ */*~ */*/*~ */*/*/*~
	cd $(DMLC_CORE); make clean; cd -
