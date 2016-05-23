#include $(CAFFE_PATH)/make/caffe.mk

# use boost
CAFFE_LDFLAGS += -lboost_system -lboost_thread -lboost_filesystem

# use gflags
CAFFE_LDFLAGS += -lgflags

# use glogs
CAFFE_LDFLAGS += -lglog

# use protobuf
CAFFE_LDFLAGS += -lprotobuf

CFLAGS += -I$(CAFFE_PATH)/include
CFLAGS += -I$(CAFFE_PATH)/build/src
LDFLAGS += $(CAFFE_LDFLAGS)

LIB_DEP += $(CAFFE_PATH)/build/lib/libcaffe.a

#$(CAFFE_PATH)/libcaffe.a:
# + cd $(CAFFE_PATH); make libcaffe.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

CAFFE_SRC = $(wildcard plugin/caffe/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(CAFFE_SRC))
CAFFE_CUSRC = $(wildcard plugin/caffe/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(CAFFE_CUSRC))
