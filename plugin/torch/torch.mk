CFLAGS += -I$(TORCH_PATH)/install/include -I$(TORCH_PATH)/install/include/TH -I$(TORCH_PATH)/install/include/THC/ -DMXNET_USE_TORCH=1
LDFLAGS += -L$(TORCH_PATH)/install/lib -lluajit -lluaT -lTH -lTHC

TORCH_SRC = $(wildcard plugin/torch/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(TORCH_SRC))
TORCH_CUSRC = $(wildcard plugin/torch/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(TORCH_CUSRC))
