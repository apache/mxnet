# Convert Caffe Model to Mxnet Format

This tool converts a caffe model into mxnet's format.

## Build
If the Caffe python package is installed then no other step is required for
using. Otherwise, it requires Google protobuf to compile Caffe's model
format. One can either install protobuf by using package manager or build from
the source. For the latter, one can set `USE_DIST_KVSTORE = 1` when compiling
mxnet, namely

```
make -C ../.. USE_DIST_KVSTORE = 1
```

Once `protobuf` is available, then run `make` in the current directory.

## How to use

Run ```python convert_model.py caffe_prototxt caffe_model save_model_name``` to convert the models. Run with ```-h``` for more details of parameters.


Or use `./run.sh model_name` to download and convert a model. Sample usage:
`./run.sh vgg19`

## Note

* We have verified the results of VGG_16 model and BVLC_googlenet results from Caffe model zoo.
* The tool only supports single input and single output network.
* The tool can only work with the L2LayerParameter in Caffe.
