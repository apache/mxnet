This example shows how to extract features with a pretrained model.

You can first get a pretrained model from <https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-bn.md>,
then prepare 2 pictures 1.jpg and 2.jpg to extract by executing `run.sh`.

Note:
1. The filename of network parameters may vary, line 67 in `feature_extract.cpp` should be updated accordingly.
2. As the build system has changed a lot, to build this example, you need to put the compiled library `libmxnet.so` in `../lib/linux`.
