TEST_SRC = $(wildcard tests/cpp/*_test.cc)
TEST = $(patsubst tests/cpp/%_test.cc, tests/cpp/%_test, $(TEST_SRC))

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

tests/cpp/%_test : tests/cpp/%_test.cc lib/libmxnet.a
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_INC) -MM -MT tests/cpp/$*_test $< >tests/cpp/$*_test.d
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_INC) -o $@ $(filter %.cc %.a, $^) $(LDFLAGS) -L$(GTEST_LIB) -lgtest

ifeq ($(USE_CUDA), 1)
# All tests for GPU in *_test_gpu.cu
TEST_SRC_GPU = $(wildcard tests/cpp/*_test_gpu.cu)
TEST_GPU = $(patsubst tests/cpp/%_test_gpu.cu, tests/cpp/%_test_gpu, $(TEST_SRC_GPU))
override TEST += $(TEST_GPU)

tests/cpp/%_test_gpu : tests/cpp/%_test_gpu.cu lib/libmxnet.a
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS) -I$(GTEST_INC)" -M -MT tests/cpp/$*_test_gpu $< >tests/cpp/$*_test_gpu.d
	$(NVCC) -c $(NVCCFLAGS) $(CUDA_ARCH) -Xcompiler "$(CFLAGS) -I$(GTEST_INC)" -o $@.o $(filter %.cu, $^)
	$(CXX) -o $@ $@.o $(filter %.a, $^) -L$(GTEST_LIB) -lgtest $(LDFLAGS) -lcudart

endif

-include tests/cpp/*.d

