# BMXNet Change Log

Note, that this is not the official mxnet changelog, but rather only the additions we made to mxnet to implement binary layers and examples for BMXNet.
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.2.0] - 2018-12-04

First beta version.
Note that binary (and quantized) examples are *maintained separately in a submodule*:
[example/bmxnet-examples](https://github.com/hpi-xnor/BMXNet-v2-examples)

### Added

- Functions:
    - `det_sign` ([ada4ea1d](https://github.com/hpi-xnor/BMXNet-v2/commit/ada4ea1d4418cfdd6cbc6d0159e1a716cb01cd85))
    - `round_ste` ([044f81f0](https://github.com/hpi-xnor/BMXNet-v2/commit/044f81f028887b9842070df28b28de394bd07516))
    - `approx_sign`
- New operator:
    - `contrib.gradcancel` (see [src/operator/contrib/gradient_cancel[-inl.h|.cc|.cu]](src/operator/contrib))
    - allows to cancel gradients (element-wise) if absolute value of input is larger than a threshold
- Binary versions of the following layers of the gluon API:
    - gluon.nn.Dense -> gluon.nn.QDense
    - gluon.nn.Conv1D -> gluon.nn.QConv1D
    - gluon.nn.Conv2D -> gluon.nn.QConv2D
    - gluon.nn.Conv3D -> gluon.nn.QConv3D
- Tests are in [tests/python/unittest/test_binary.py](tests/python/unittest/test_binary.py)
- Layers are in [python/mxnet/gluon/nn/binary_layers.py](python/mxnet/gluon/nn/binary_layers.py)

### Changed

- Code in [python/mxnet/visualization.py](python/mxnet/visualization.py) changed:
    - `plot_network` skips certain layers which clutter binary network graphs
    - `print_summary` calculates compressed model size and prints number of binarized/quantized and full-precision weights

## [0.1.0] - 2018-09-05

Version of BMXNet started (coincides with adapting our code to Gluon).
