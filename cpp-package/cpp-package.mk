ifndef LINT_LANG
	LINT_LANG="all"
endif

ifdef CAFFE_PATH
export LD_LIBRARY_PATH=$(CAFFE_PATH)/lib
endif

CPP_PACKAGE_OP_H_FILE = cpp-package/include/mxnet-cpp/op.h

EXTRA_PACKAGES += cpp-package-all
EXTRA_PACKAGES_CLEAN += cpp-package-clean

.PHONY: cpp-package-all cpp-package-lint cpp-package-clean

cpp-package-all: $(CPP_PACKAGE_OP_H_FILE)

cpp-package-clean:
	rm -f $(CPP_PACKAGE_OP_H_FILE)

$(CPP_PACKAGE_OP_H_FILE): lib/libmxnet.so cpp-package/scripts/OpWrapperGenerator.py
	(cd cpp-package/scripts; python OpWrapperGenerator.py $(ROOTDIR)/lib/libmxnet.so)

cpp-package-lint:
	(cd cpp-package; python scripts/lint.py dmlc ${LINT_LANG} include example)

include cpp-package/example/example.mk

