CPPP_MAKE_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))
CPPP_THISDIR := $(dir $(CPPP_MAKE_FILE))

include $(CPPP_THISDIR)/example/example.mk

ifndef LINT_LANG
	LINT_LANG="all"
endif

export LD_LIBRARY_PATH=$(CAFFE_PATH)/lib

.PHONY += cpp-package-all cpp-package-lint cpp-package-clean

cpp-package-all: $(CPPP_THISDIR)/include/mxnet-cpp/op.h

cpp-package-clean: cpp-package-example-clean
	rm -f $(CPPP_THISDIR)/include/mxnet-cpp/op.h

$(CPPP_THISDIR)/include/mxnet-cpp/op.h: $(CPPP_THISDIR)/src/OpWrapperGenerator/OpWrapperGenerator.py $(CPPP_THISDIR)/../lib/libmxnet.so
	(cd $(CPPP_THISDIR)/src/OpWrapperGenerator; python OpWrapperGenerator.py $(ROOTDIR)/lib/libmxnet.so)

cpp-package-lint:
	(cd $(CPPP_THISDIR); python $(CPPP_THISDIR)/scripts/lint.py dmlc ${LINT_LANG} include example)


