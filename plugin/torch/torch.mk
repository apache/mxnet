CFLAGS += -I$(TORCH_PATH)/install/include -I$(TORCH_PATH)/install/include/TH -I$(TORCH_PATH)/install/include/THC/ -DMXNET_USE_TORCH=1
LDFLAGS += -L$(TORCH_PATH)/install/lib -lluajit -lluaT -lTH -lTHC -L$(TORCH_PATH)/install/lib/lua/5.1 -lpaths -ltorch -lnn
ifeq ($(USE_CUDA), 1)
	LDFLAGS += -lcutorch -lcunn
endif

TORCH_SRC = $(wildcard plugin/torch/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(TORCH_SRC))
TORCH_CUSRC = $(wildcard plugin/torch/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(TORCH_CUSRC))