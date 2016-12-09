all: caffe_parse/caffe_pb2.py

clean:
	rm caffe_parse/caffe_pb2.py*

caffe_parse/caffe_pb2.py:
	protoc --python_out=./ ./caffe_parse/caffe.proto
