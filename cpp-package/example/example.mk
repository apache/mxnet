CPPEX_SRC = $(wildcard cpp-package/example/*.cpp)
CPPEX_EXE = $(patsubst cpp-package/example/%.cpp, build/cpp-package/example/%, $(CPPEX_SRC))

CPPEX_CFLAGS += -Icpp-package/include -Ibuild/cpp-package/include
CPPEX_EXTRA_LDFLAGS := -L$(ROOTDIR)/lib -lmxnet

EXTRA_PACKAGES += cpp-package-example-all
EXTRA_PACKAGES_CLEAN += cpp-package-example-clean

.PHONY: cpp-package-example-all cpp-package-example-clean

cpp-package-example-all: cpp-package-all $(CPPEX_EXE)

build/cpp-package/example/% : cpp-package/example/%.cpp lib/libmxnet.so $(CPP_PACKAGE_OP_H_FILE)
	@mkdir -p $(@D)
	$(CXX) -std=c++0x $(CFLAGS) $(CPPEX_CFLAGS) -MM -MT cpp-package/example/$* $< >build/cpp-package/example//$*.d
	$(CXX) -std=c++0x $(CFLAGS) $(CPPEX_CFLAGS) -o $@ $(filter %.cpp %.a, $^) $(LDFLAGS) $(CPPEX_EXTRA_LDFLAGS)

cpp-package-example-clean:
	rm -rf build/cpp-package/example/*

-include build/cpp-package/example/*.d
