CFLAGS += -I$(CAFFE_PATH)/include -I$(CAFFE_PATH)/build/src -I$(CAFFE_PATH)/build/include
LDFLAGS += -lprotobuf -lboost_system -lboost_thread -lboost_filesystem -lgflags -lglog -L$(CAFFE_PATH)/build/lib -lcaffe

ifeq ($(USE_CUDNN), 1)
	CFLAGS += -DUSE_CUDNN=1
endif

ifeq ($(USE_CUDA), 0)
	CFLAGS += -DCPU_ONLY=1
endif

CAFFE_SRC = $(wildcard plugin/caffe/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(CAFFE_SRC))
CAFFE_CUSRC = $(wildcard plugin/caffe/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(CAFFE_CUSRC))
