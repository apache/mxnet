CFLAGS += -I$(WARPCTC_PATH)/include
LDFLAGS += -L$(WARPCTC_PATH)/build -lwarpctc

WARPCTC_SRC = $(wildcard plugin/warpctc/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(WARPCTC_SRC))
WARPCTC_CUSRC = $(wildcard plugin/warpctc/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(WARPCTC_CUSRC))
