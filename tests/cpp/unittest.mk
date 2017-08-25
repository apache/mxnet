TEST_SRC = $(shell find tests/cpp/ -name "*.cc")
TEST_OBJ = $(patsubst %.cc, build/%.o, $(TEST_SRC))
TEST = build/tests/cpp/mxnet_test

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

TEST_CFLAGS = -Itests/cpp/include -Isrc $(CFLAGS)
TEST_LDFLAGS = $(LDFLAGS) -Llib -lmxnet

ifeq ($(USE_BREAKPAD), 1)
TEST_CFLAGS  += -I/usr/local/include/breakpad
TEST_LDFLAGS += -lbreakpad_client -lbreakpad
endif

.PHONY: runtest testclean

build/tests/cpp/%.o : tests/cpp/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(TEST_CFLAGS) -MM -MT tests/cpp/$* $< > build/tests/cpp/$*.d
	$(CXX) -c -std=c++0x $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/$*.o $(filter %.cc %.a, $^)

build/tests/cpp/operator/%.o : tests/cpp/operator/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(TEST_CFLAGS) -MM -MT tests/cpp/operator/$* $< > build/tests/cpp/operator/$*.d
	$(CXX) -c -std=c++0x $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/operator/$*.o $(filter %.cc %.a, $^)

build/tests/cpp/storage/%.o : tests/cpp/storage/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(TEST_CFLAGS) -MM -MT tests/cpp/storage/$* $< > build/tests/cpp/storage/$*.d
	$(CXX) -c -std=c++0x $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/storage/$*.o $(filter %.cc %.a, $^)

build/tests/cpp/engine/%.o : tests/cpp/engine/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(TEST_CFLAGS) -MM -MT tests/cpp/engine/$* $< > build/tests/cpp/engine/$*.d
	$(CXX) -c -std=c++0x $(TEST_CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/engine/$*.o $(filter %.cc %.a, $^)

$(TEST): $(TEST_OBJ) lib/libmxnet.so
	$(CXX) -std=c++0x $(TEST_CFLAGS) -I$(GTEST_INC) -o $@ $^ $(TEST_LDFLAGS) -L$(GTEST_LIB) -lgtest

runtest: $(TEST)
	LD_LIBRARY_PATH=$(shell pwd)/lib:$(LD_LIBRARY_PATH) $(TEST)

testclean:
	rm -f $(TEST) $(TEST_OBJ)

-include build/tests/cpp/*.d
-include build/tests/cpp/operator/*.d
-include build/tests/cpp/storage/*.d
-include build/tests/cpp/engine/*.d