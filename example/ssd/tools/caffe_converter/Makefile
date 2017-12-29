ifndef PROTOC
DEPS_PROTOC=../../deps/bin/protoc
ifneq ("$(wildcard $(DEPS_PROTOC))","")
PROTOC = $(DEPS_PROTOC)
else
PROTOC = protoc
endif
endif

all: caffe_pb2.py

clean:
	rm caffe_pb2.py*

caffe_pb2.py:
	$(PROTOC) --python_out=./ ./caffe.proto
