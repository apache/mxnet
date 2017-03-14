TEST_SRC = $(wildcard tests/cpp/*.cc)
TEST_OBJ = $(patsubst %.cc, build/%.o, $(TEST_SRC))
TEST = build/tests/cpp/mxnet_test

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

build/tests/cpp/%.o : tests/cpp/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT tests/cpp/$* $< > build/tests/cpp/$*.d
	$(CXX) -c -std=c++0x $(CFLAGS) -I$(GTEST_INC) -o build/tests/cpp/$*.o $(filter %.cc %.a, $^)

$(TEST): $(TEST_OBJ) lib/libmxnet.a
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_INC) -o $@ $^ $(LDFLAGS) -L$(GTEST_LIB) -lgtest

-include build/tests/cpp/*.d
