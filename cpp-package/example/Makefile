CPPEX_SRC = $(wildcard *.cpp)
CPPEX_EXE = $(patsubst %.cpp, %, $(CPPEX_SRC))

CFLAGS += -I../../include -I../../nnvm/include -I../../dmlc-core/include
CPPEX_CFLAGS += -I../include
CPPEX_EXTRA_LDFLAGS := -L. -lmxnet

.PHONY: all clean

all: $(CPPEX_EXE)

$(CPPEX_EXE):% : %.cpp libmxnet.so ../include/mxnet-cpp/*.h
	$(CXX) -std=c++0x $(CFLAGS) $(CPPEX_CFLAGS) -o $@ $(filter %.cpp %.a, $^) $(CPPEX_EXTRA_LDFLAGS)

clean:
	rm -f $(CPPEX_EXE)
