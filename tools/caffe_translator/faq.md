### Frequently asked questions

[**Why is Caffe required to run the translated code?**](#why_caffe)

There is a couple of reasons why Caffe is required to run the translated code:

1. The translator does not convert Caffe data layer to native MXNet code because MXNet cannot read from LMDB files. Translator instead generates code that uses [`CaffeDataIter`](https://mxnet.incubator.apache.org/faq/caffe.html#use-io-caffedataiter) which can read LMDB files. `CaffeDataIter` needs Caffe to run.

2. If the Caffe code to be translated uses custom layers, or layers that don't have equivalent MXNet layers, the translator will generate code that will use [CaffeOp](https://mxnet.incubator.apache.org/faq/caffe.html#use-sym-caffeop). CaffeOp needs Caffe to run.

[**What version of Caffe prototxt can the translator translate?**](#what_version_of_prototxt)

Caffe Translator supports the `proto2` syntax.

[**Can the translator translate Caffe 2 code?**](#caffe_2_support)

No. At the moment, only Caffe is supported.
