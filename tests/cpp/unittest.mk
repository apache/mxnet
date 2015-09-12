UNITTEST_SRC = $(wildcard tests/cpp/*_unittest.cc)
UNITTEST_OBJ = $(patsubst tests/cpp/%_unittest.cc, tests/cpp/%_unittest.o, $(UNITTEST_SRC))

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

tests/cpp/%.o : tests/cpp/%.cc 
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT tests/$*.o $< >tests/$*.d
	$(CXX) -std=c++0x -c $(CFLAGS) -I$(GTEST_INC) -c $< -o $@

tests/cpp/unittest: $(UNITTEST_OBJ) lib/libmxnet.a
	$(CXX) $(CFLAGS) -std=c++0x  -o $@ $(filter %.o %.a, $^)  $(LDFLAGS) -lgtest -lgtest_main 

-include tests/cpp/*.d


