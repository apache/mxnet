CPPEXMAKE_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))
CPPEX_THISDIR := $(dir $(CPPEXMAKE_FILE))

CPPEX_SRC = $(wildcard cpp-package/example/*.cpp)
CPPEX_EXE = $(patsubst cpp-package/example/%.cpp, build/cpp-package/example/%, $(CPPEX_SRC))

CFLAGS += -I$(CPPEX_THISDIR)/../include
EXTRA_LDFLAGS := -L$(ROOTDIR)/lib -lmxnet

.PHONY += cpp-package-example-all cpp-package-example-clean

BIN += $(CPP_EXE)

cpp-package-example-all: $(CPPEX_EXE)

build/cpp-package/example/% : cpp-package/example/%.cpp
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_INC) -MM -MT cpp-package/example/$* $< >build/cpp-package/example//$*.d
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_INC) -o $@ $(filter %.cpp %.a, $^) $(LDFLAGS) $(EXTRA_LDFLAGS)

cpp-package-example-clean:
	-rm -f build/cpp-package/example/*

-include build/cpp-package/example/*.d
