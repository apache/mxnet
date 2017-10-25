# Frequently asked questions

[**- Why is Caffe needed to run the translated code?**](#why_caffe)

There is a couple of reasons why Caffe is needed:

1. The translator does not convert caffe data layer to native MXNet code because MXNet cannot read from LMDB files. Translator instead generates code that uses CaffeDataIter which can read LMDB files. CaffeDataIter needs Caffe to run.

2. If the caffe code to be translated uses custom layers, or layers that don't have equivalent MXNet layers (like scale layer), translator will generate code that will use CaffeOp. CaffeOp needs Caffe to run.
