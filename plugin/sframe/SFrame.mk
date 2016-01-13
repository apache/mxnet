SFRMAE_SRC = plugin/sframe/iter_sframe.cc
PLUGIN_OBJ += build/plugin/sframe/iter_sframe.o
CFLAGS += -I$(SFRAME_PATH)/oss_src/unity/lib/
CFLAGS += -I$(SFRAME_PATH)/oss_src/
LDFLAGS += -L$(SFRAME_PATH)/release/oss_src/unity/python/sframe/
LDFLAGS += -lunity_shared
LDFLAGS += -lboost_system
