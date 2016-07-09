CFLAGS += -I$(CAFFE_PATH)/include -I$(CAFFE_PATH)/build/src
LDFLAGS += -lprotobuf -lboost_system -lboost_thread -lboost_filesystem -lgflags -lglog -L$(CAFFE_PATH)/build/lib -lcaffe

CAFFE_SRC = $(wildcard plugin/caffe/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(CAFFE_SRC))
CAFFE_CUSRC = $(wildcard plugin/caffe/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(CAFFE_CUSRC))
