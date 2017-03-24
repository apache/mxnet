ifndef config
ifdef CXXNET_CONFIG
  config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ../../config.mk)","")
  config = ../config.mk
else
  config = ../make/config.mk
endif
endif

include $(config)

ifndef LINT_LANG
	LINT_LANG="all"
endif

include ../config.mk

.PHONY: all lint example clean

all: include/mxnet-cpp/op.h example

clean:
	rm -f include/mxnet-cpp/op.h

include/mxnet-cpp/op.h: src/OpWrapperGenerator/OpWrapperGenerator.py ../lib/libmxnet.so
	(cd src/OpWrapperGenerator; python OpWrapperGenerator.py ../../../lib/libmxnet.so)

lint:
	python scripts/lint.py dmlc ${LINT_LANG} include example

example:
	$(MAKE) -C example travis

