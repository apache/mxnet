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

OBJ = storage.o

LIB_DEP = $(DMLC_CORE)/libdmlc.a

.PHONY: clean all

all: $(OBJ)

$(DMLC_CORE)/libdmlc.a:
	+ cd $(DMLC_CORE); make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

storage.o: src/storage/storage.cc


$(BIN) :
	$(CXX) $(CFLAGS)  -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(OBJCXX11) :
	$(CXX) -std=c++0x -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(SLIB) :
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(OBJCXX11) $(BIN) $(CUBIN) $(CUOBJ) $(SLIB) *~ */*~ */*/*~
	cd $(DMLC_CORE); make clean; cd -
