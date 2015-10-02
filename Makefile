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

ifneq ($(USE_OPENMP), 1)
	export NO_OPENMP = 1
endif

# use customized config file
include $(config)
include mshadow/make/mshadow.mk
include $(DMLC_CORE)/make/dmlc.mk
unexport NO_OPENMP

# all tge possible warning tread
WARNFLAGS= -Wall
CFLAGS = -DMSHADOW_FORCE_STREAM $(WARNFLAGS)

# CFLAGS for debug
ifeq ($(DEBUG), 1)
	CFLAGS += -g -O0
else
	CFLAGS += -O3
endif
CFLAGS += -I./mshadow/ -I./dmlc-core/include -fPIC -Iinclude $(MSHADOW_CFLAGS)
LDFLAGS = -pthread $(MSHADOW_LDFLAGS) $(DMLC_LDFLAGS)
NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)
ROOTDIR = $(CURDIR)

ifndef LINT_LANG
	LINT_LANG="all"
endif

# setup opencv
ifeq ($(USE_OPENCV), 1)
	CFLAGS += -DMXNET_USE_OPENCV=1 `pkg-config --cflags opencv`
	LDFLAGS += `pkg-config --libs opencv`
	BIN += bin/im2rec
else
	CFLAGS+= -DMXNET_USE_OPENCV=0
endif

ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
endif

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DMSHADOW_USE_CUDNN=1
	LDFLAGS += -lcudnn
endif

ifeq ($(USE_THREADED_ENGINE), 1)
	CFLAGS += -DMXNET_USE_THREADED_ENGINE
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

ifneq ($(USE_CUDA_PATH), NONE)
	NVCC=$(USE_CUDA_PATH)/bin/nvcc
endif

.PHONY: clean all test lint doc clean_all

all: lib/libmxnet.a lib/libmxnet.so $(BIN)

SRC = $(wildcard src/*.cc src/*/*.cc)
OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
CUSRC = $(wildcard src/*/*.cu)
CUOBJ = $(patsubst src/%.cu, build/%_gpu.o, $(CUSRC))

LIB_DEP = $(DMLC_CORE)/libdmlc.a
ALL_DEP = $(OBJ) $(LIB_DEP)
ifeq ($(USE_CUDA), 1)
	ALL_DEP += $(CUOBJ)
endif

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -std=c++0x -c $(CFLAGS) -c $< -o $@

build/%_gpu.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -M build/$*_gpu.o $< >build/$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $<

lib/libmxnet.a: $(ALL_DEP)
	ar crv $@ $(filter %.o, $?)

lib/libmxnet.so: $(ALL_DEP)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

bin/im2rec: tools/im2rec.cc $(DMLC_CORE)/libdmlc.a

$(BIN) :
	$(CXX) $(CFLAGS)  -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

include tests/cpp/unittest.mk

test: $(TEST)

lint:
	python dmlc-core/scripts/lint.py mxnet ${LINT_LANG} include src scripts python

doxygen:
	doxygen doc/Doxyfile

clean:
	$(RM) -r build lib/lib* *~ */*~ */*/*~ */*/*/*~

clean_all: clean
	cd $(DMLC_CORE); make clean; cd -

-include build/*.d
-include build/*/*.d
