TEST_SRC = $(wildcard tests/cpp/*_test.cc)
TEST = $(patsubst tests/cpp/%_test.cc, tests/cpp/%_test, $(TEST_SRC))

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

tests/cpp/% : tests/cpp/%.cc lib/libmxnet.a
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT tests/cpp/$* $< >tests/cpp/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_INC) -o $@ $(filter %.cc %.a, $^) $(LDFLAGS) -L$(GTEST_LIB) -lgtest

-include tests/cpp/*.d
