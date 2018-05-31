This example shows how to extract features with a pretrained model.

Execute `run.sh` to:
- Download a pretrained model
- Download sample pictures (`1.jpg` and `2.jpg`)
- Compile the files
- Execute the featurization on `1.jpg` and `2.jpg`

Note:
1. The filename of network parameters may vary, line 67 in `feature_extract.cpp` should be updated accordingly.
2. You need to build MXNet from source to get access to the `lib/libmxnet.so`
