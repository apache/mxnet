OPENCV_SRC = $(wildcard plugin/opencv/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(OPENCV_SRC))
OPENCV_CUSRC = $(wildcard plugin/opencv/*.cu)
PLUGIN_CUOBJ += $(patsubst %.cu, build/%_gpu.o, $(OPENCV_CUSRC))
