This example shows how to extract features with a pretrained model.

Execute `run.sh` to:
- Download a pretrained model
- Download sample pictures (`dog.jpg` and `cat.jpg`)
- Compile the files
- Execute the featurization on `dog.jpg` and `cat.jpg`


Note:
1. The filename of network parameters may vary, line 67 in `feature_extract.cpp` should be updated accordingly.
2. You need to build MXNet from source to get access to the `lib/libmxnet.so` or point `LD_LIBRARY_PATH` to where it is installed in your system
