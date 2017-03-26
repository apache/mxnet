CPPEXMAKE_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))
CPPEX_THISDIR := $(dir $(CPPEXMAKE_FILE))

CFLAGS += -I$(CPPEX_THISDIR)/../include
EXTRA_LDFLAGS := -L$(ROOTDIR)/lib -lmxnet

.PHONY += cpp-package-example-all cpp-package-example-clean temp

cpp-package-example-all: cpp-package-all mlp lenet lenet_with_mxdataiter alexnet googlenet resnet

temp:
	echo "CPPEXMAKE_FILE: " $(CPPEXMAKE_FILE)
	echo "CPPEX_THISDIR: " $(CPPEX_THISDIR)

lenet_with_mxdataiter: $(CPPEX_THISDIR)/lenet_with_mxdataiter.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)
	-rm -f $(basename $@).o

lenet: $(CPPEX_THISDIR)/lenet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)
	-rm -f $(basename $@).o

mlp: $(CPPEX_THISDIR)/mlp.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)
	-rm -f $(basename $@).o

alexnet: $(CPPEX_THISDIR)/alexnet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)
	-rm -f $(basename $@).o

googlenet: $(CPPEX_THISDIR)/googlenet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)
	-rm -f $(basename $@).o

inception_bn: $(CPPEX_THISDIR)/inception_bn.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)
	-rm -f $(basename $@).o

resnet: $(CPPEX_THISDIR)/resnet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS) $(EXTRA_LDFLAGS)
	-rm -f $(basename $@).o

# For simplicity, no link here
#cpp-package-example-travis:
#	$(CXX) -c -std=c++11 $(CFLAGS) ./mlp.cpp && rm -f mlp.o
#	$(CXX) -c -std=c++11 $(CFLAGS) ./lenet.cpp && rm -f lenet.o
#	$(CXX) -c -std=c++11 $(CFLAGS) ./lenet_with_mxdataiter.cpp && rm -f lenet_with_mxdataiter.o
#	$(CXX) -c -std=c++11 $(CFLAGS) ./alexnet.cpp && rm -f alexnet.o
#	$(CXX) -c -std=c++11 $(CFLAGS) ./googlenet.cpp && rm -f googlenet.o
#	$(CXX) -c -std=c++11 $(CFLAGS) ./resnet.cpp && rm -f resnet.o


cpp-package-example-clean:
	-rm -f mlp
	-rm -f lenet
	-rm -f lenet_with_mxdataiter
	-rm -f alexnet
	-rm -f googlenet
	-rm -f resnet
	-rm -f inception_bn
