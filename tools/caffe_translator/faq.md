# Frequently asked questions

[**Why is Caffe needed to run the translated code?**](#why_caffe)

There is a couple of reasons why Caffe is needed:

1. The translator does not convert caffe data layer to native MXNet code because MXNet cannot read from LMDB files. Translator instead generates code that uses CaffeDataIter which can read LMDB files. CaffeDataIter needs Caffe to run.

2. If the caffe code to be translated uses custom layers, or layers that don't have equivalent MXNet layers (like scale layer), translator will generate code that will use CaffeOp. CaffeOp needs Caffe to run.

[**Which Caffe layers can the translator automatically translate?**](#supported_layers)

- Accuracy and Top-k
- Batch Normalization
- Concat
- Convolution
- Data<sup>*</sup>
- Deconvolution
- Eltwise
- Inner Product (Fully Connected layer)
- Flatten
- Permute
- Pooling
- Power
- Relu
- Scale<sup>*</sup>
- SoftmaxOutput

<sup>*</sup> - Uses [CaffePlugin](https://github.com/apache/incubator-mxnet/tree/master/plugin/caffe)
